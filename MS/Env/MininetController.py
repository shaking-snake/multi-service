import os
import subprocess
from time import sleep, time
import re # 导入正则表达式库
from mininet.net import Mininet
from enum import Enum # 需要导入 FlowGenerator 中的 Enum
from contextlib import contextmanager
import torch
import sys
import signal
import shlex
import networkx as nx
from mininet.topo import Topo
from functools import partial

from scapy.all import rdpcap
from scapy.layers.inet import IP
from .FlowGenerator import FlowType, FLOW_PROFILES
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink

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

# 生成一个mininet网络

# mininet 定义
class GraphTopo(Topo):
  def __init__(self, blueprint_g: nx.Graph, r2q_value: int, **opts):
    Topo.__init__(self, **opts)

    for node_id in blueprint_g.nodes():
      self.addSwitch(f's{node_id}', protocols='OpenFlow13')
      self.addHost(f'h{node_id}')
      self.addLink(f'h{node_id}', f's{node_id}', delay='0.1ms')

    for u, v, data in blueprint_g.edges(data=True):
      bw = data.get('bandwidth', 1000)
      delay = f"{data.get('delay', 1)}ms"
      # 这里沿用 Mininet 构造函数中设置的 r2q
      self.addLink(f's{u}', f's{v}', delay=delay) 

# mininet 启动
@contextmanager
def get_a_mininet(g: nx.Graph):
  RemoteCtrl = partial(RemoteController, ip='127.0.0.1', port=6633)
  NEW_R2Q = 80000 # 解决 HTB 警告

  net = Mininet(
    topo=GraphTopo(g, r2q_value=NEW_R2Q),
    switch=OVSKernelSwitch,
    link=TCLink,
    controller=RemoteCtrl,
  )

  try:
    net.start()
    yield net
  finally:
    net.stop()
    os.system('sudo mn -c')

  return net

# 发送流量并捕获包特征
def send_packet_and_capture(
  server, 
  client, 
  flow_type: FlowType, 
  duration_sec=15, 
  n_packets_to_capture=30, 
  **flow_params
  ):
  # 发送流并抓包
  """
  在 Mininet 中运行 D-ITG 流量, 并同时使用 tshark 管道实时捕获特征。

  参数:
    net: Mininet 网络对象。
    flow_type (str): 'voip', 'gaming', 'streaming'.
    duration_sec (int): D-ITG 流量的*总*运行时长。
    n_packets_to_capture (int): tshark 在捕获 N 个包后自动停止。
    **flow_params: 传递给 generate_ditg_command 的额外参数。
  
  返回:
    np.ndarray: 形状为 (N, 3) 的特征矩阵 [[Size, IAT], ...]。
  """

  server_ip = server.IP()
  client_ip = client.IP()
  
  print(f"server ip: {server_ip} client_ip: {client_ip}")
  # 2. 找到要监听的接口 (s1-eth1)
  server_intf = None
  for intf in server.intfList():
    if intf.name != 'lo' and intf.link: # 确保它不是 'lo' 并且已连接
      server_intf = intf
      break
  if server_intf is None:
    raise Exception(f"在 {server.name} 上找不到已连接的数据接口!")

  switch_intf = server_intf.link.intf2 if server_intf.link.intf1 == server_intf else server_intf.link.intf1
  switch_intf_name = switch_intf.name
  
  # 3. [Action 1] 获取 D-ITG 命令
  client_cmd = get_flow_command(
    flow_type=flow_type,
    target_ip=server_ip,
    duration_sec=duration_sec,
    **flow_params
  )

  # 4. 准备 tshark 命令 (这是最快的方法)
  display_filter = f"src host {client_ip} and dst host {server_ip}"
  tshark_cmd = [
    'sudo',
    'tshark',
    '-c', str(n_packets_to_capture), # 抓 N 个包后停止
    '-i', switch_intf_name,
    '-l', # 行缓冲 (实时)
    '-T', 'fields',
    '-e', 'frame.len',        # 特征 1: Size
    '-e', 'frame.time_delta', # 特征 2: IAT
    '-e', 'ip.src',
    '-e', 'ip.dst',
    '-E', 'separator=,',
    '-f', display_filter
  ]
  
  feature_matrix = []
  client_proc = None
  tshark_proc = None
  server_proc = None

  try:
    # 5. [Action 2] 启动流量
    print(f"[Net] 启动 D-ITG 接收端 (h1)...")
    server_proc = server.popen('ITGRecv')
    sleep(1)
  
    # 6. [Action 3] 启动 tshark 捕获管道
    print(f"[Capture] 启动 tshark 管道: {' '.join(tshark_cmd)}")
    tshark_proc = subprocess.Popen(
      tshark_cmd, 
      stdout=subprocess.PIPE, 
      stderr=subprocess.PIPE,
      text=True
    )

    sleep(1)
    print(f"[Net] 启动 D-ITG 发送端 (h2): {client_cmd}")
    client_proc = client.popen(client_cmd)
    # 7. [核心] 实时从管道读取并封装向量
    for line in tshark_proc.stdout:
      line = line.strip()
      if not line:
        continue
      print(f"[RAW CAPTURE] 抓到了: {line}")
      try:

        size_str, iat_str, src_ip, dst_ip = line.split(',')
        
        size = float(size_str)
        
        # 处理第一个包 (IAT 不是数字)
        try:
          iat = float(iat_str)
        except ValueError:
          iat = 0.0 # 第一个包的 IAT 为 0
        
        # 实时封装成向量
        feature_vector = [size, iat]
        feature_matrix.append(feature_vector)
      except ValueError as e:
        print(f"[Parser] 跳过 tshark 行: {line}. 错误: {e}")
      
  except Exception as e:
    print(f"[Error] 实验执行出错: {e}")
  finally:
    # 8. 清理
    print("[Net] 清理进程...")
    if tshark_proc: tshark_proc.terminate()
    if client_proc: client_proc.terminate()
    if server_proc: server_proc.terminate()

  print(f"[Capture] 捕获完成. 获得 {len(feature_matrix)} 个向量。")
  return torch.tensor(feature_matrix, dtype=float)

# 根据流类型，返回不同的 D-ITG 命令。
def get_flow_command(
  flow_type: str, 
  target_ip: str, 
  duration_sec: int, 
  **kwargs
  ) -> str:
    """
    根据流量模式和参数, 生成一个 D-ITG (ITGSend) 命令字符串。
    所有命令均使用 D-ITG (ITGSend) 工具。

    参数:
      flow_type (str): 流量模式。支持: 'voip', 'gaming', 'streaming'.
      target_ip (str): 目标服务器的 IP 地址 (例如: '10.0.0.1').
      duration_sec (int): 流量的总持续时间 (秒).
    
    返回:
      str: 一个完整的、可在 Mininet 主机上运行的 ITGSend 命令字符串。
    """
    profile = FLOW_PROFILES[flow_type]

    protocol = profile['protocol']
    duration_ms = duration_sec * 1000
    
    # 基础命令 (所有模式通用), 明确使用 D-ITG 的 ITGSend
    base_cmd = f"ITGSend -a {shlex.quote(target_ip)} -t {duration_ms} -T {protocol}"
    
    if 'ditg_preset' in profile:
      # 使用预设 (VoIP, Gaming)
      specific_args = profile['ditg_preset']
    elif 'ditg_manual' in profile:
      # 使用手动参数 (Streaming)
      specific_args = profile['ditg_manual']
    else:
      raise ValueError("Profile 配置不完整")
   

    # 组合命令, 并在末尾添加 '&' 使其在后台运行
    final_cmd = f"{base_cmd} {specific_args}"
    
    return final_cmd





