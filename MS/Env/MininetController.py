import os
import subprocess
import time
import re # 导入正则表达式库
from mininet.net import Mininet
from enum import Enum # 需要导入 FlowGenerator 中的 Enum
import numpy as np
import sys
import signal

from scapy.all import rdpcap
from scapy.layers.inet import IP
from .FlowGenerator import FlowType, FLOW_PROFILES

def parse_iperf_output(output_str: str, mode: str) -> dict:
  """
  【新增】解析 iperf 的原始字符串输出。
  """
  metrics = {'bandwidth': 0.0, 'jitter': 0.0, 'loss_rate': 1.0}
  
  try:
    if mode == 'udp':
      # UDP 模式: 关注 Jitter 和 Loss
      # 示例: 9.99 Mbits/sec  0.013 ms    0/ 1429 (0%)
      udp_regex = re.search(
        r"(\d+\.?\d*)\s+Mbits/sec\s+(\d+\.?\d*)\s+ms\s+\d+/\s+\d+\s+\((\d+\.?\d*)%\)", 
        output_str
      )
      if udp_regex:
        metrics['bandwidth'] = float(udp_regex.group(1))
        metrics['jitter'] = float(udp_regex.group(2))
        metrics['loss_rate'] = float(udp_regex.group(3)) / 100.0 # 转换为 0.0 - 1.0
            
    elif mode == 'tcp':
      # TCP 模式: 关注 Bandwidth
      # 示例: 8.99 Mbits/sec
      tcp_regex = re.search(
        r"(\d+\.?\d*)\s+Mbits/sec", 
        output_str
      )
      if tcp_regex:
        # TCP 几乎没有丢包（因为它会重传），Jitter 不作为主要考量
        metrics['bandwidth'] = float(tcp_regex.group(1))
        metrics['loss_rate'] = 0.0 
        metrics['jitter'] = 0.0
              
  except Exception as e:
    print(f"Error parsing iperf output: {e}\nOutput: {output_str}")
      
  return metrics

def measure_latency_ping(S_host, D_host, num_packets=10) -> float:
    """
    【新增】使用 ping 测量平均延迟 (RTT/2)。
    iperf 不擅长测量延迟，因此我们独立测量。
    """
    try:
      result = S_host.cmd(f'ping -c {num_packets} {D_host.IP()}')
      # 解析 ping 的统计摘要
      # 示例: rtt min/avg/max/mdev = 0.050/0.065/0.081/0.012 ms
      ping_regex = re.search(r"rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/", result)
      
      if ping_regex:
        avg_rtt = float(ping_regex.group(1))
        return avg_rtt / 2.0 # 单向延迟估计
            
    except Exception as e:
      print(f"Error measuring ping: {e}\nOutput: {result}")
        
    return 1000.0 # 返回一个惩罚性的高延迟

def calculate_qoe_reward(qos_metrics: dict, flow_profile: dict) -> float:
  """
  【新增】根据 QoE “悬崖”计算奖励。
  这是 RL 训练信号的核心。
  """
  
  # 默认奖励为正，表示成功
  reward = 10.0 
  
  # 获取 QoE 关键阈值
  qoe_reqs = flow_profile.get('qoe_critical', {})
  
  # 1. 检查延迟 (来自 Ping)
  max_delay = qoe_reqs.get('max_delay')
  if max_delay is not None and qos_metrics.get('delay', 1000.0) > max_delay:
    print(f"REWARD PENALTY: Delay cliff! {qos_metrics.get('delay')} > {max_delay}")
    return -100.0 # 巨大的惩罚
      
  # 2. 检查抖动 (来自 iperf -u)
  max_jitter = qoe_reqs.get('max_jitter')
  if max_jitter is not None and qos_metrics.get('jitter', 1000.0) > max_jitter:
    print(f"REWARD PENALTY: Jitter cliff! {qos_metrics.get('jitter')} > {max_jitter}")
    return -100.0 # 巨大的惩罚

  # 3. 检查带宽 (来自 iperf -c 或 -u)
  min_bw = qoe_reqs.get('min_bandwidth')
  if min_bw is not None and qos_metrics.get('bandwidth', 0.0) < min_bw:
    print(f"REWARD PENALTY: Bandwidth cliff! {qos_metrics.get('bandwidth')} < {min_bw}")
    return -100.0 # 巨大的惩罚
      
  # 4. 检查丢包率 (来自 iperf -u)
  max_loss = qoe_reqs.get('max_loss_rate')
  if max_loss is not None and qos_metrics.get('loss_rate', 1.0) > max_loss:
    print(f"REWARD PENALTY: Loss cliff! {qos_metrics.get('loss_rate')} > {max_loss}")
    return -100.0 # 巨大的惩罚

  # 如果通过了所有悬崖测试，可以返回一个更精细的奖励
  # (例如，VoIP 可以基于 E-Model 计算 R-factor)
  # 简单起见，我们先返回一个固定的正奖励
  # TODO
  return reward

def measure_path_qos(S_host, D_host, path_route, flow_profile):
  """
  【已完善】测量给定路径和流量的 QoS 并计算奖励。
  这是 RL Env 的核心 step() 函数。
  """
  
  # 1. 配置 OpenFlow 规则 (将流量强制导向 path_route)
  # ---------------------------------------------------------------
  # TODO: 这是你的“动作”应用点
  # 你的 RL Agent (Actor) 需要输出一个 'path_route'
  # 你需要在这里实现一个函数，根据 'path_route' 列表（例如 [s1, s3, s4, d1]）
  # 来安装 OpenFlow 规则，强制 S_host 和 D_host 之间的流量
  # 必须经过这条路径。
  # 
  # e.g., setup_flow_rules(ovs_switches, path_route)
  # ---------------------------------------------------------------
  print(f"INFO: Simulating path (TODO: implement OVS rules)...")

  # 2. 【改进】并行测量延迟 (Ping) 和 吞吐/抖动 (iperf)
  
  # 2.A 启动 iperf server
  mode = flow_profile['iperf_mode']
  D_host.cmd(f'iperf -s -p 5002 -i 1 -m &') # -m: 打印 MTU 和头部信息

  # 2.B 启动 ping 测量延迟
  # 注意：我们在 iperf 运行时测量 ping，以捕获“负载下”的延迟
  print("INFO: Starting parallel Ping measurement...")
  ping_result_future = S_host.popen(f'ping -c 5 -i 0.2 {D_host.IP()}', shell=True)
  
  # 3. 运行 iperf client
  rate = flow_profile['target_rate']
  duration = 5 # 模拟持续 5 秒
  
  client_cmd = f'iperf -c {D_host.IP()} -p 5002 -{mode[0]} -b {rate} -t {duration} -f M' # -f M: 统一单位为 Mbits/sec
  if mode == 'udp':
      client_cmd += " -l 160" # VoIP 模拟: 160B 负载 (G.711 20ms 帧)
      
  print(f"INFO: Starting iperf client ({mode})...")
  iperf_result_str = S_host.cmd(client_cmd)
  
  # 4. 清理 server 进程
  D_host.cmd('kill %iperf')
  
  # 5. 解析结果
  
  # 5.A 解析 iperf (带宽, 抖动, 丢包)
  print("INFO: Parsing iperf results...")
  qos_metrics = parse_iperf_output(iperf_result_str, mode)
  
  # 5.B 解析 ping (延迟)
  print("INFO: Parsing ping results...")
  ping_output = ping_result_future.communicate()[0].decode()
  qos_metrics['delay'] = measure_latency_ping_from_output(ping_output) # (见下方辅助函数)
  
  print(f"--- QOS METRICS ---")
  print(qos_metrics)
  print(f"-------------------")
  
  # 6. 计算 Reward
  reward = calculate_qoe_reward(qos_metrics, flow_profile)
  
  print(f"=== REWARD ===: {reward}")
  
  return reward

def measure_latency_ping_from_output(result: str) -> float:
  """
  【辅助】从 ping 字符串中解析平均延迟。
  """
  try:
    ping_regex = re.search(r"rtt min/avg/max/mdev = [\d\.]+/([\d\.]+)/", result)
    if ping_regex:
      avg_rtt = float(ping_regex.group(1))
      return avg_rtt / 2.0 # 单向延迟
  except Exception as e:
    print(f"Error parsing ping output: {e}")
  return 1000.0 # 惩罚值


# Generator :

# 获取流指纹
def get_flow_fingerprint(S_host, D_host, flow_profile):
    
  # 运行捕获
  pcap_path = run_traffic_capture(S_host, D_host, flow_profile)
  
  if not os.path.exists(pcap_path):
    print(f"⚠️ 捕获文件未找到: {pcap_path}")
    return None
      
  # 使用 Scapy 读取文件
  fingerprint_matrix = extract_flow_fingerprint(pcap_path)
  
  # 清理临时文件 (重要!)
  os.remove(pcap_path)
  
  return fingerprint_matrix

# 从.pcap文件中获取流指纹
def extract_flow_fingerprint(pcap_path: str, n_packets: int = 50) -> np.ndarray:
  """
  使用 Scapy 从 .pcap 文件中提取流指纹 (PacketSize, IAT)。

  :param pcap_path: .pcap 文件的路径。
  :param n_packets: 要提取的最大数据包数量 (N)。
  :return: 一个 (n_packets, 2) 的 NumPy 矩阵。
  """
  
  # 1. 初始化
  features_list = []
  last_timestamp = None
  
  try:
    # 2. 使用 Scapy 读取 pcap 文件
    # rdpcap() 将整个文件读入内存中的一个 PacketList
    packets = rdpcap(pcap_path)

    # 3. 循环遍历数据包
    for packet in packets:
      
      # 4. 过滤：我们只关心 IP 层的数据包
      if IP not in packet:
        continue
          
      # 5. 特征提取
      try:
        # 特征 1: PacketSize (IP 层的总长度)
        packet_size = float(packet[IP].len)
        
        # 特征 2: IAT (Inter-Arrival Time)
        current_timestamp = float(packet.time)
        
        if last_timestamp is None:
          # 第一个包的 IAT 设为 0
          iat = 0.0
        else:
          iat = current_timestamp - last_timestamp
            
        last_timestamp = current_timestamp
        
        # 6. 存储特征
        features_list.append([packet_size, iat])
        
        # 7. 检查是否达到 N 个包
        if len(features_list) >= n_packets:
          break
    
      except Exception as e:
        # 处理 Scapy 解析数据包时可能发生的罕见错误
        print(f"警告: 解析数据包时出错: {e}", file=sys.stderr)
              
  except Exception as e:
    print(f"错误: 无法读取 pcap 文件 {pcap_path}: {e}", file=sys.stderr)
    # 如果 pcap 文件损坏或不存在，返回一个全零矩阵
    return np.zeros((n_packets, 2), dtype=np.float32)

  # 8. 转换为 NumPy 矩阵
  num_found = len(features_list)
  
  if num_found == 0:
    # 如果没有找到任何 IP 包，返回全零矩阵
    return np.zeros((n_packets, 2), dtype=np.float32)
      
  matrix = np.array(features_list, dtype=np.float32)
  
  # 9. 填充 (Padding)
  # 如果捕获的包少于 n_packets (例如流很短)
  if num_found < n_packets:
    # 创建一个 (n_packets - num_found, 2) 的零矩阵用于填充
    padding = np.zeros((n_packets - num_found, 2), dtype=np.float32)
    # 垂直堆叠
    matrix = np.vstack([matrix, padding])
      
  return matrix

# 发送流并获取.pcap 文件
def run_traffic_capture(S_host, D_host, flow_profile, N_PACKETS=50):
    
  temp_pcap_file = f"/tmp/fingerprint_{os.getpid()}.pcap"
  mode = flow_profile['iperf_mode']
  rate = flow_profile['target_rate']
  
  # --- 根据模式启动对应的服务器 ---
  iperf_server_cmd = f'iperf -s -p 5001'
  if mode == 'udp':
    iperf_server_cmd += ' -u'
  
  D_host.popen(iperf_server_cmd)

  correct_intf = None
  for intf in S_host.intfNames():
    if intf != 'lo' :
      correct_intf = intf
      break
  
  if correct_intf is None:
    print(f"错误: 找不到 {S_host.name} 的有效接口 (非 'lo')")
    D_host.cmd('kill %iperf')
    return None

  capture_command = f'tcpdump -i {correct_intf} -c {N_PACKETS} -w {temp_pcap_file}'

  # B. 使用 Popen 启动 tcpdump
  tcpdump_proc = S_host.popen(capture_command, shell=True)

  # C. 等待 tcpdump 启动
  time.sleep(0.5) 

  # D. 启动客户机，发送流
  client_cmd = f'iperf -c {D_host.IP()} -p 5001 -{mode[0]} -b {rate} -t 60'
  client_proc = S_host.popen(client_cmd, shell=True) 

  tcpdump_proc.wait()
  client_proc.terminate()
  
  return temp_pcap_file

