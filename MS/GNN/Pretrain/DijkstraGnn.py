import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import networkx as nx
import math
from torch.utils.data import IterableDataset

# === 全局统计 (保持不变) ===
GLOBAL_STATS = {
  'delay': {'min': 1.0, 'max': 10.0},
  'bw':    {'min': 100.0, 'max': 1000.0},
  'degree': {'min': 2.0, 'max': 10.0}
}

# === 核心函数 1: 图数据转换 (含 5 维节点特征) ===
def get_pyg_data_from_nx(G: nx.Graph, S_node: int, D_node: int, global_stats: dict):
  """
  将 NetworkX 图转换为 PyG Data，关键在于注入 BFS 距离特征。
  """
  # --- 1. 边特征 (Edge Attributes) ---
  source_nodes, target_nodes = [], []
  edge_attrs_raw = []
  for u, v, data in G.edges(data=True):
    # 无向图，添加双向边
    source_nodes.extend([u, v])
    target_nodes.extend([v, u])
    # 原始 delay 和 bw
    attr = [data.get('delay', 1.0), data.get('bandwidth', 1000.0)]
    edge_attrs_raw.extend([attr, attr])

  edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
  edge_attr_tensor = torch.tensor(edge_attrs_raw, dtype=torch.float)

  # 归一化边特征
  d_min, d_max = global_stats['delay']['min'], global_stats['delay']['max']
  b_min, b_max = global_stats['bw']['min'], global_stats['bw']['max']
  edge_attr = torch.zeros_like(edge_attr_tensor)
  edge_attr[:, 0] = (edge_attr_tensor[:, 0] - d_min) / (d_max - d_min + 1e-6)
  edge_attr[:, 1] = (edge_attr_tensor[:, 1] - b_min) / (b_max - b_min + 1e-6)
  edge_attr = edge_attr.clamp(0.0, 1.0)

  # --- 2. 节点特征 (Node Features) [关键: 5维特征] ---
  # 计算 BFS 跳数
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
  deg_max = global_stats['degree']['max']

  for i in range(num_nodes):
      # Feat 1: 归一化度
      deg = G.degree(i) / (deg_max + 1e-6)
      # Feat 2 & 3: 源/目的标记
      is_s = 1.0 if i == S_node else 0.0
      is_d = 1.0 if i == D_node else 0.0
      # Feat 4 & 5: 归一化 BFS 距离 (假设最大直径约 20)
      ds = min(dist_from_s.get(i, 999), 20) / 20.0
      dd = min(dist_to_d.get(i, 999), 20) / 20.0
      
      node_features_list.append([deg, is_s, is_d, ds, dd])

  x = torch.tensor(node_features_list, dtype=torch.float)

  return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), G

# === 核心函数 2: 专家标签生成 (保持不变) ===
def generate_expert_label(G: nx.Graph, S_node: int, D_node: int, edge_index: torch.Tensor):
  try:
    path_nodes = nx.dijkstra_path(G, S_node, D_node, weight='delay')
  except nx.NetworkXNoPath:
    return None

  path_edges = set()
  for i in range(len(path_nodes) - 1):
    u, v = path_nodes[i], path_nodes[i+1]
    path_edges.add((u, v))
    path_edges.add((v, u)) 

  num_total_edges = edge_index.shape[1]
  labels = torch.zeros(num_total_edges, dtype=torch.float)
  for i in range(num_total_edges):
    u, v = edge_index[0, i].item(), edge_index[1, i].item()
    if (u, v) in path_edges:
      labels[i] = 1.0
  return labels

# === 核心类 1: 动态数据集 (已正确实现) ===
def generate_single_sample(topo_gen, global_stats):
    while True:
        G_nx = topo_gen.generate_topology()
        try:
            S, D = topo_gen.select_source_destination()
        except: continue
        data, G_with_attrs = get_pyg_data_from_nx(G_nx, S, D, global_stats)
        y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
        if y is not None:
            data.y = y
            return data

class DynamicGraphDataset(IterableDataset):
    def __init__(self, topo_gen, global_stats, max_samples):
        self.topo_gen = topo_gen
        self.stats = global_stats
        self.max_samples = max_samples
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_end = self.max_samples
        else:
            per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
            iter_end = min(worker_info.id * per_worker + per_worker, self.max_samples)
        for _ in range(iter_end):
            yield generate_single_sample(self.topo_gen, self.stats)

# === 核心类 2: GNN 模型 (含边属性感知头) ===
class GNNPretrainModel(nn.Module):
  def __init__(self, node_feat_dim, gnn_dim, edge_feat_dim, num_layers=6):
    super().__init__()
    self.gnn_dim = gnn_dim
    self.num_layers = num_layers
    
    self.node_embed = nn.Linear(node_feat_dim, gnn_dim)
    self.convs = nn.ModuleList()
    self.layer_norms = nn.ModuleList()

    for _ in range(num_layers):
      self.layer_norms.append(nn.LayerNorm(gnn_dim))
      self.convs.append(
        pyg_nn.GATConv(gnn_dim, gnn_dim, heads=2, concat=False, edge_dim=edge_feat_dim)
      )
      
    # [关键: 输出头输入维度 = gnn_dim*2 + edge_feat_dim]
    self.edge_output_head = nn.Sequential(
      nn.Linear(gnn_dim * 2 + edge_feat_dim, gnn_dim),
      nn.ReLU(),
      nn.Linear(gnn_dim, 1)
    )

  def forward(self, data, manual_gamma, manual_beta):
    h = self.node_embed(data.x)
    for l in range(self.num_layers):
      h_norm = self.layer_norms[l](h)
      h_modulated = manual_gamma[l].unsqueeze(0) * h_norm + manual_beta[l].unsqueeze(0)
      h = self.convs[l](h_modulated, data.edge_index, edge_attr=data.edge_attr)
      h = h.relu()
      
    # [关键: 拼接边属性]
    h_src = h[data.edge_index[0]]
    h_dst = h[data.edge_index[1]]
    edge_features = torch.cat([h_src, h_dst, data.edge_attr], dim=-1)
    
    return self.edge_output_head(edge_features).squeeze()