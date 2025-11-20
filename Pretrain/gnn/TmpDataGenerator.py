import torch
import math
import numpy as np
import random
import networkx as nx
from torch.utils.data import IterableDataset

# 引入必要的辅助函数
# 确保 NetworkGenerator.py 里有 get_pyg_data_from_nx
from MS.Env.NetworkGenerator import get_pyg_data_from_nx  

# ==========================================
# 1. 独立的样本生成函数 (逻辑解耦)
# ==========================================
def generate_expert_label(G, S_node, D_node, edge_index):
  """
  (辅助函数) 生成 Dijkstra 标签
  """
  try:
    path_nodes = nx.dijkstra_path(G, S_node, D_node, weight='delay')
  except nx.NetworkXNoPath:
    return None

  path_edges = set()
  for i in range(len(path_nodes) - 1):
    u, v = path_nodes[i], path_nodes[i+1]
    path_edges.add((u, v))
    path_edges.add((v, u)) # 无向图双向都要标

  num_total_edges = edge_index.shape[1]
  labels = torch.zeros(num_total_edges, dtype=torch.float)
  for i in range(num_total_edges):
    u, v = edge_index[0, i].item(), edge_index[1, i].item()
    if (u, v) in path_edges:
      labels[i] = 1.0
  return labels

def generate_single_sample(topo_gen, config):
  """
  生成单个样本的原子操作。
  放在类外面，避免 pickle 问题，也方便单独测试。
  """
  while True:
    # 1. 生成拓扑
    G_nx = topo_gen.generate_topology()
    
    try:
      # 2. 随机选点
      S, D = topo_gen.select_source_destination()
      
      # 3. 提取特征 (调用唯一的真理函数)
      # 注意：这里必须从 NetworkGenerator 导入 get_pyg_data_from_nx
      data, G_with_attrs = get_pyg_data_from_nx(G_nx, S, D, config)
      
      # 4. 生成标签
      y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
      
      if y is not None:
        data.y = y
        return data
              
    except Exception as e:
      # 捕获所有异常（如连通性问题、特征提取失败），重试
      # print(f"样本生成失败，重试: {e}")
      print(e)
      continue

# ==========================================
# 2. Dataset 类 (只负责调度)
# ==========================================
class DynamicGraphDataset(IterableDataset):
  def __init__(self, topo_gen, config, max_samples):
    self.topo_gen = topo_gen
    self.config = config
    self.max_samples = max_samples

  def __iter__(self):
    # --- 多进程 Worker 配置 ---
    worker_info = torch.utils.data.get_worker_info()
    
    if worker_info is None:
      # 单进程模式
      iter_start = 0
      iter_end = self.max_samples
    else:
      # 多进程模式：每个 Worker 负责一部分数据
      # 1. 设置随机数种子，防止所有 Worker 生成一样的图
      worker_seed = torch.initial_seed() % (2**32 - 1) + worker_info.id
      random.seed(worker_seed)
      np.random.seed(worker_seed)
      
      # 2. 计算该 Worker 需要生成的数量
      per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
      worker_id = worker_info.id
      
      iter_start = worker_id * per_worker
      iter_end = min(iter_start + per_worker, self.max_samples) # [修正] 这里原来写成了 self.config

    # --- 生成循环 ---
    # 计算这个 Worker 实际要生成的数量
    count = iter_end - iter_start
    
    for _ in range(count):
      yield generate_single_sample(self.topo_gen, self.config)