import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical
from torch_geometric.nn.glob import global_mean_pool

# === 导入自定义模块 ===
# 请确保项目结构正确，并且 __init__.py 文件存在
from MS.A2C.ActorCritic import ActorCritic
from MS.Env.MininetController import get_a_mininet, get_a_fingerprint, measure_path_qos
from MS.Env.FlowGenerator import FlowGenerator
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx

# ================= 配置参数 =================
class Config:
  # --- 拓扑生成参数 ---
  #
  M_BA = 2
  MIN_BW = 100.0
  MAX_BW = 1000.0
  MIN_DELAY = 1.0
  MAX_DELAY = 10.0
  MIN_NODES_NUM = 15
  MAX_NODES_NUM = 30

  # --- 训练控制 ---
  MAX_EPISODES = 2000      # 总回合数
  EPISODES_PER_TOPO = 64   # 每套拓扑训练多少个流 (复用次数)
  
  # --- 优化器参数 ---
  LR = 1e-4                # 学习率 (Phase 2 微调需要较小 LR)
  GAMMA = 0.99             # 折扣因子
  ENTROPY_COEF = 0.01      # 熵正则化 (鼓励探索)
  MAX_GRAD_NORM = 0.5      # 梯度裁剪阈值
  CRITIC_LOSS_COEF = 0.5   # Value Loss 权重

  # --- 路径配置 ---
  MODEL_DIR = "./trained_model"
  SAVE_PATH = os.path.join(MODEL_DIR, "a2c_final.pth")
  # 确保这些预训练模型文件存在
  PRETRAINED_LSTM = os.path.join(MODEL_DIR, "trained_lstm.pth") 
  PRETRAINED_GNN = os.path.join(MODEL_DIR, "gnn_pretraned_model.pth") # 注意文件名拼写是否与您实际一致

  # --- 模型结构 (需与定义保持一致) ---
  GNN_DIM = 256
  LSTM_DIM = 128
  GNN_LAYERS = 6
  
  # --- 环境参数 ---
  N_PACKETS = 30           # 抓包数量 (指纹长度)
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = Config()

# ================= 辅助函数 1: 路径采样 =================

def sample_path_probabilistic(edge_logits, edge_index, s_node, d_node, max_steps=20):
  """
  根据 GNN 输出的边分数 (Logits)，概率性地游走采样一条路径。
  返回: (path_nodes, log_prob_sum, is_success)
  """
  # 构建邻接表以便快速查询: u -> [(v, edge_idx_in_tensor), ...]
  adj = {}
  num_edges = edge_index.shape[1]
  for i in range(num_edges):
    u = edge_index[0, i].item()
    v = edge_index[1, i].item()
    if u not in adj: adj[u] = []
    adj[u].append((v, i))

  current = s_node
  path = [current]
  log_probs = []
  visited = {current}
    
  for _ in range(max_steps):
    if current == d_node:
      break # 到达终点
        
    if current not in adj:
      break # 死胡同
        
    # 获取合法邻居 (不走回头路)
    neighbors = adj[current]
    valid_options = [n for n in neighbors if n[0] not in visited]

    if not valid_options:
      break # 无路可走
        
    # 提取候选边的 logits
    candidate_logits = []
    candidate_nodes = []
    
    for next_node, edge_idx in valid_options:
      candidate_logits.append(edge_logits[edge_idx])
      candidate_nodes.append(next_node)
        
    # 转换为局部概率分布 (Softmax)
    logits_tensor = torch.stack(candidate_logits)
    probs = torch.softmax(logits_tensor, dim=0)
    
    # 采样动作
    dist = Categorical(probs)
    action_idx = dist.sample()
    
    # 记录
    log_prob = dist.log_prob(action_idx)
    log_probs.append(log_prob)
    
    next_hop = candidate_nodes[action_idx.item()]
    path.append(next_hop)
    visited.add(next_hop)
    current = next_hop
        
  is_success = (path[-1] == d_node)
  
  # 汇总整条路径的 Log Probability
  if len(log_probs) > 0:
    total_log_prob = torch.stack(log_probs).sum()
  else:
    total_log_prob = torch.tensor(0.0, device=edge_logits.device)
      
  return path, total_log_prob, is_success

# ================= 辅助函数 2: 流表操作 =================

def clean_flow_rules(net, cookie=0x1234):
    """
    清理指定 cookie 的流表规则，避免上一回合的残留规则干扰。
    """
    for sw in net.switches:
        # 删除带有特定 cookie 的规则 (strict matching)
        # 格式: ovs-ofctl del-flows <switch> "cookie=0x1234/-1"
        # /-1 表示掩码，匹配该 cookie 的所有规则
        sw.cmd(f'ovs-ofctl -O OpenFlow13 del-flows {sw.name} "cookie={cookie}/-1"')

def install_path_rules(net, path_nodes, cookie=0x1234):
  """
  将逻辑路径转化为 OpenFlow 流表规则并下发。
  支持双向联通 (TCP/ARP 需要回路)，或者仅单向 (UDP)。
  
  参数:
    net: Mininet 网络对象
    path_nodes: 节点 ID 列表，例如 [0, 2, 5] 代表 s0 -> s2 -> s5
    cookie: 规则标记，用于清理
  """
  # 1. 获取源主机和目的主机对象
  # 根据 GraphTopo 的命名规则: 节点 i -> 主机 hi, 交换机 si
  src_id = path_nodes[0]
  dst_id = path_nodes[-1]
  
  h_src = net.get(f'h{src_id}')
  h_dst = net.get(f'h{dst_id}')
  dst_ip = h_dst.IP()
  src_ip = h_src.IP()
  
  print(f"[Controller] Installing path: h{src_id} -> ... -> h{dst_id}")

  # 2. 遍历路径上的每一跳交换机
  for i, current_node_id in enumerate(path_nodes):
    sw_name = f's{current_node_id}'
    switch = net.get(sw_name)
    
    # --- 确定输出端口 (Out Port) ---
    if i == len(path_nodes) - 1:
      # Case A: 最后一跳 (s_dst)，要去往主机 h_dst
      # 获取 switch 连接到 h_dst 的端口
      # linksBetween 返回 [(intf1, intf2)], 我们需要 switch 侧的 intf
      links = net.linksBetween(switch, h_dst)
      if not links: continue
      link = links[0]
      # 确定哪个接口属于 switch
      out_intf = link.intf1 if link.intf1.node == switch else link.intf2
      out_port = switch.ports[out_intf]
    else:
      # Case B: 中间跳，要去往下一跳交换机 s_next
      next_node_id = path_nodes[i+1]
      next_switch = net.get(f's{next_node_id}')
      
      links = net.linksBetween(switch, next_switch)
      if not links: continue
      link = links[0]
      out_intf = link.intf1 if link.intf1.node == switch else link.intf2
      out_port = switch.ports[out_intf]

    # --- 下发规则 (Forwarding) ---
    # 匹配: 目的 IP 是 dst_ip
    # 动作: 转发到 out_port
    # 优先级: 100 (高于默认规则)
    cmd = (
      f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
      f'"cookie={cookie},priority=100,dl_type=0x0800,nw_dst={dst_ip},actions=output:{out_port}"'
    )
    switch.cmd(cmd)
    
    # --- [重要] 下发反向规则 (Reverse Path) ---
    # TCP 握手和 iperf 结束报告需要回包。
    # 简单起见，我们让反向流量沿原路返回。
    if i == 0:
      # 第一跳的反向出口是去往 h_src
      links_rev = net.linksBetween(switch, h_src)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]
    else:
      # 中间跳的反向出口是去往上一跳交换机 s_prev
      prev_node_id = path_nodes[i-1]
      prev_switch = net.get(f's{prev_node_id}')
      links_rev = net.linksBetween(switch, prev_switch)
      if links_rev:
        link_rev = links_rev[0]
        rev_intf = link_rev.intf1 if link_rev.intf1.node == switch else link_rev.intf2
        rev_port = switch.ports[rev_intf]
    
    # 反向规则: 匹配目的 IP 是 src_ip (即回包)
    cmd_rev = (
      f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} '
      f'"cookie={cookie},priority=100,dl_type=0x0800,nw_dst={src_ip},actions=output:{rev_port}"'
    )
    switch.cmd(cmd_rev)

    # --- [可选] 处理 ARP ---
    # 如果没有 Controller 处理 ARP，需要广播 ARP 请求
    # 动作: output:FLOOD (或者 normal 如果有默认学习规则)
    switch.cmd(f'ovs-ofctl -O OpenFlow13 add-flow {sw_name} "cookie={cookie},priority=100,dl_type=0x0806,actions=FLOOD"')

# ================= 主训练循环 =================

def run_a2c_training():
  # 1. 初始化模型
  print(f"[ms] 正在初始化 A2C Agent (Device: {CONFIG.DEVICE})...")
  agent = ActorCritic(
    lstm_hidden_dim=CONFIG.LSTM_DIM,
    gnn_hidden_dim=CONFIG.GNN_DIM,
    gnn_layers=CONFIG.GNN_LAYERS,
    pretrained_lstm_path=CONFIG.PRETRAINED_LSTM,
    pretrained_gnn_path=CONFIG.PRETRAINED_GNN).to(CONFIG.DEVICE)
    
  agent.train() # 开启训练模式 (主要是 Dropout/BatchNorm, 虽然这里用得少)
  
  # 优化器 (主要训练 FiLM 和新 Heads)
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, agent.parameters()), lr=CONFIG.LR)
  
  # 2. 初始化生成器
  topo_gen = TopologyGenerator(CONFIG)
  flow_gen = FlowGenerator()
  
  # 计算循环次数
  num_topo_updates = CONFIG.MAX_EPISODES // CONFIG.EPISODES_PER_TOPO
  total_episodes = 0
  stats_reward = []

  print(f"[ms] 开始训练: 共 {num_topo_updates} 次拓扑变更, 总计 {CONFIG.MAX_EPISODES} 回合")
  
  # --- 外层循环：拓扑生命周期 ---
  for topo_idx in range(num_topo_updates):
      
    # A. 生成新拓扑
    G_nx = topo_gen.generate_topology()
    print(f"[Topo {topo_idx+1}/{num_topo_updates}] 启动 Mininet (Nodes: {len(G_nx.nodes())})...")
    
    try:
      # B. 启动 Mininet (Context Manager)
      with get_a_mininet(G_nx) as net:
        # 预取节点对象
        hosts = {i: net.get(f'h{i}') for i in G_nx.nodes()}
        
        # 等待网络稳定
        # net.pingAll() # 可选，确保连通性
        
        # --- 内层循环：流训练 (Mininet 复用) ---
        pbar = tqdm(range(CONFIG.EPISODES_PER_TOPO), desc=f"Topo {topo_idx+1}")
        
        for _ in pbar:
          total_episodes += 1
          
          # 1. 环境准备
          clean_flow_rules(net) # 清理上一回合规则
          s_node, d_node = topo_gen.select_source_destination()
          h_src, h_dst = hosts[s_node], hosts[d_node]
          
          # 2. 获取状态 (State)
          # 2.1 采集指纹 (耗时操作)
          flow_type, flow_profile = flow_gen.get_random_flow()
          fingerprint = get_a_fingerprint(
            server=h_dst, client=h_src, 
            flow_type=flow_type, 
            n_packets_to_capture=CONFIG.N_PACKETS).float().to(CONFIG.DEVICE) # Shape: (1, N, 2)
          
          # 2.2 获取图特征
          pyg_data, _ = get_pyg_data_from_nx(G_nx, s_node, d_node, CONFIG)
          pyg_data = pyg_data.to(CONFIG.DEVICE)
            
          # 3. 模型推理 (Manual Forward 以获取中间变量)
          dist, value_est, edge_logits = agent(fingerprint, pyg_data)

          # 4. 动作采样 (Action)
          path, log_prob_sum, success = sample_path_probabilistic(
            edge_logits, pyg_data.edge_index, s_node, d_node)
          
          # 5. 执行与奖励 (Reward)
          if not success:
            reward = -10.0
          else:
            install_path_rules(net, path)
            # 测量真实 QoS
            reward = measure_path_qos(h_src, h_dst, path, flow_profile)
          
          # 归一化奖励 (简单除以常数，防止梯度过大)
          # reward_norm = reward / 10.0 
          reward_tensor = torch.tensor([reward], device=CONFIG.DEVICE)
          
          # 6. 反向传播 (Update)
          # Advantage
          advantage = reward_tensor - value_est.detach()
          
          # Losses
          actor_loss = -log_prob_sum * advantage
          critic_loss = nn.MSELoss()(value_est, reward_tensor)
          
          total_loss = actor_loss + CONFIG.CRITIC_LOSS_COEF * critic_loss
          
          optimizer.zero_grad()
          total_loss.backward()
          torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG.MAX_GRAD_NORM)
          optimizer.step()
          
          # 7. 记录与保存
          stats_reward.append(reward)
          avg_r = np.mean(stats_reward[-50:])
          pbar.set_postfix({"R": f"{reward:.1f}", "Avg": f"{avg_r:.1f}", "L": f"{total_loss.item():.2f}"})
          
          if total_episodes % 100 == 0:
            torch.save(agent.state_dict(), CONFIG.SAVE_PATH)

    except Exception as e:
      print(f"\n[Error] Topo {topo_idx} 异常中断: {e}")
      import traceback
      traceback.print_exc()
      continue # 尝试下一个拓扑，而不是直接退出

  print(f"训练结束。最终模型已保存至 {CONFIG.SAVE_PATH}")

if __name__ == '__main__':
    if os.getuid() != 0:
    print("错误: Mininet 必须以 root 权限运行 (sudo python3 ...)")
  else:
    os.makedirs(CONFIG.MODEL_DIR, exist_ok=True)
    # 确保日志目录存在 (如果 MininetController 需要写日志)
    os.makedirs("./train-log", exist_ok=True)
    run_a2c_training()
