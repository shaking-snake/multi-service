import networkx as nx
import random
import torch
from torch_geometric.data import Data

class Default_config:
  # 默认拓扑生成参数
  M_BA = 2

  # mbps
  MIN_BW = 100.0
  MAX_BW = 1000.0

  # %
  MIN_LOSS = 0.0 
  MIN_LOSS = 10.0

  # ms
  MIN_DELAY = 1.0 
  MAX_DELAY = 200.0

  # 
  MIN_NODES_NUM = 15
  MAX_NODES_NUM = 30

DEFAULT_CONFIG = Default_config()

class TopologyGenerator:
  def __init__(self, config=DEFAULT_CONFIG):
    """
    初始化拓扑生成器。
    :param num_nodes_range: 节点数量的范围 (e.g., 10-20个节点)
    :param m_ba: BA模型的附着参数 (每个新节点连接到 m_ba 个现有节点)
    """
    self.num_nodes_range = (config.MIN_NODES_NUM, config.MAX_NODES_NUM)
    self.m_ba = config.M_BA 
    self.min_bw = config.MIN_BW  # Mbps
    self.max_bw = config.MAX_BW  # Mbps
    self.min_loss = config.MIN_LOSS
    self.max_loss = config.MAX_LOSS
    self.min_delay = config.MIN_DELAY # ms
    self.max_delay = config.MAX_DELAY # ms
    self.G = None

  def generate_topology(self) -> nx.Graph:
    """生成一个随机的BA拓扑，并赋予链路属性。"""
    num_nodes = random.randint(*self.num_nodes_range)
    
    # 1. 生成 BA 无标度网络
    G = nx.barabasi_albert_graph(n=num_nodes, m=self.m_ba)
    
    # 2. 为节点和边添加属性
    for u, v in G.edges():
      # 随机但合理的链路属性
      bw = random.uniform(self.min_bw, self.max_bw)
      delay = random.uniform(self.min_delay, self.max_delay)
      loss = random.uniform(self.min_loss, self.min_loss)
      # 存储属性
      G[u][v]['bandwidth'] = bw
      G[u][v]['delay'] = delay
      G[u][v]['loss'] = loss
      # 存储容量（例如，基于带宽或QCI等级）
      G[u][v]['capacity'] = bw * 0.8 # 假设可用容量是带宽的80%

    self.G = G
    return G

  def select_source_destination(self) -> tuple[int, int]:
    """随机选择不重复的源和目的节点。"""
    if self.G is None:
      raise ValueError("Topology must be generated first.")
        
    nodes = list(self.G.nodes())
    
    # 确保源和目的一定是连通的
    while True:
      s, d = random.sample(nodes, 2)
      if nx.has_path(self.G, s, d):
        return s, d

def get_pyg_data_from_nx(G: nx.Graph, S_node: int, D_node: int, config):
  # 1. 预计算图结构特征
  try:
    # 对于小图(10-30节点)，这些计算非常快
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, alpha=0.85)
  except:
    # 容错处理
    betweenness = {n: 0.0 for n in G.nodes()}
    clustering = {n: 0.0 for n in G.nodes()}  
    pagerank = {n: 0.0 for n in G.nodes()}

  source_nodes, target_nodes = [], []
  edge_attrs_raw = []
  
  for u, v, data in G.edges(data=True):
    source_nodes.extend([u, v])
    target_nodes.extend([v, u])
    
    # --- [关键修改] 边特征扩展为 4 维 ---
    d = data.get('delay', 1.0)
    l = data.get('loss', 0.0)       # 丢包率
    b = data.get('bandwidth', 1000.0)
    u_load = data.get('utilization', 0.0) # 利用率 (预训练默认为0)

    attr = [d, b, l, u_load]
    edge_attrs_raw.extend([attr, attr])

  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long) 
  edge_attr_tensor = torch.tensor(edge_attrs_raw, dtype=torch.float)

  # 归一化
  edge_attr = torch.zeros_like(edge_attr_tensor)
  # Delay
  edge_attr[:, 0] = (edge_attr_tensor[:, 0] - config.MIN_DELAY) / (config.MAX_DELAY - config.MIN_DELAY + 1e-6)
  # Bandwidth
  edge_attr[:, 1] = (edge_attr_tensor[:, 1] - config.MIN_BW) / (config.MAX_BW - config.MIN_BW + 1e-6)
  # Loss & Utilization (本身就是 0-1)
  edge_attr[:, 2] = edge_attr_tensor[:, 2]
  edge_attr[:, 3] = edge_attr_tensor[:, 3]
  
  edge_attr = edge_attr.clamp(0.0, 1.0)

  # 距离计算
  try:
    dist_from_s = nx.single_source_shortest_path_length(G, S_node)
  except:
    dist_from_s = {n: 999 for n in G.nodes()}

  try:
    dist_to_d = nx.single_source_shortest_path_length(G, D_node)
  except:
    dist_to_d = {n: 999 for n in G.nodes()}

  num_nodes = G.number_of_nodes()
  node_features_list = []
  deg_max = config.MAX_NODES_NUM # 使用配置中的最大节点数作为归一化分母

  for i in range(num_nodes):
    # --- [关键修改] 节点特征扩展为 8 维 ---
    deg = G.degree(i) / (deg_max + 1e-6)
    is_s = 1.0 if i == S_node else 0.0
    is_d = 1.0 if i == D_node else 0.0
    # 距离特征
    ds = 1.0 * min(dist_from_s.get(i, 999), deg_max) / deg_max 
    dd = 1.0 * min(dist_to_d.get(i, 999), deg_max) / deg_max
    # 结构特征
    betw = betweenness.get(i, 0.0)
    clus = clustering.get(i, 0.0)
    pr = pagerank.get(i, 0.0) * 10.0 # 适当放大 PageRank

  node_features_list.append([deg, is_s, is_d, ds, dd, betw, clus, pr])

  x = torch.tensor(node_features_list, dtype=torch.float)
  return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), G

