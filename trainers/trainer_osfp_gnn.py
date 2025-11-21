import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Batch
from torch.utils.data import IterableDataset
from tqdm import tqdm
import os
import math
import numpy as np
import random
import networkx as nx
import torch.nn.functional as F  # [关键修复] 补全 F 导入

# === 导入自定义模块 ===
from MS.GNN.FiLMGnn import FiLMGnn 
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx

# === 辅助函数 ===
def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  
  # [修复 Barrier 警告 1] 先设置当前进程可见的 GPU
  torch.cuda.set_device(rank)
  
  # 初始化进程组
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_expert_label(G, S_node, D_node, edge_index):
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

# === 配置 ===
class Config:
    MIN_NODES_NUM = 15
    MAX_NODES_NUM = 30
    M_BA = 2
    MIN_BW = 2.0
    MAX_BW = 30.0
    MIN_DELAY = 1.0
    MAX_DELAY = 200.0
    MIN_LOSS = 0.0
    MAX_LOSS = 2.0

# === 动态数据集 ===
def generate_single_sample(topo_gen, config):
    while True:
        try:
            G_nx = topo_gen.generate_topology()
            S, D = topo_gen.select_source_destination()
            data, G_with_attrs = get_pyg_data_from_nx(G_nx, S, D, config)
            y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
            if y is not None:
                data.y = y
                return data
        except Exception:
            continue

class DynamicGraphDataset(IterableDataset):
    def __init__(self, topo_gen, config, max_samples_per_epoch, rank, world_size):
        self.topo_gen = topo_gen
        self.config = config
        self.samples_per_gpu = int(math.ceil(max_samples_per_epoch / world_size))
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        base_seed = torch.initial_seed() 
        if worker_info is not None:
            base_seed += worker_info.id
        
        # 加上 Rank 偏移
        unique_seed = base_seed + (self.rank * 10000)
        
        # [修正 Numpy 种子溢出] 限制种子范围
        random.seed(unique_seed)
        np.random.seed(unique_seed % (2**32 - 1)) 
        
        if worker_info is None:
            iter_count = self.samples_per_gpu
        else:
            per_worker = int(math.ceil(self.samples_per_gpu / float(worker_info.num_workers)))
            iter_count = min(per_worker, max(0, self.samples_per_gpu - worker_info.id * per_worker))
            
        for _ in range(iter_count):
            yield generate_single_sample(self.topo_gen, self.config)

# === 损失函数 ===
class FocalLoss(nn.Module):
  def __init__(self, alpha=0.85, gamma=2.0, logits=True, reduce=True):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.logits = logits
    self.reduce = reduce

  def forward(self, inputs, targets):
    if self.logits:
      BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    else:
      BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)
    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
    if self.reduce: return torch.mean(F_loss)
    else: return F_loss

# === 训练主逻辑 ===
def train_worker(rank, world_size):
  try:
    # 1. 初始化
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
      print(f"🚀 启动 DDP 训练 | GPU: {world_size} | PID: {os.getpid()}")

    # 2. 参数
    EPOCHS = 1000          
    BATCH_SIZE = 128       
    TOTAL_SAMPLES = 6400   
    LEARNING_RATE = 5e-4
    
    # [确认这里的维度与 NetworkGenerator 一致]
    NODE_FEAT_DIM = 10  # 10维 (含 Buffer, ProcDelay)
    EDGE_FEAT_DIM = 5   # 5维 (含 AvailBW)
    GNN_DIM = 256
    NUM_LAYERS = 6
    
    SAVE_PATH = "./trained_model/gnn_pretrained_model.pth"
    if rank == 0:
      os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    # 3. 模型
    model = FiLMGnn(
      node_feat_dim=NODE_FEAT_DIM, 
      gnn_dim=GNN_DIM, 
      edge_feat_dim=EDGE_FEAT_DIM, 
      num_layers=NUM_LAYERS
    ).to(device)
    
    # [DDP 性能优化] find_unused_parameters=False 消除警告并加速
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    loss_fn = FocalLoss(alpha=0.85, gamma=2.0)

    best_acc = 0.0
    topo_gen = TopologyGenerator(Config)

    # 4. 循环
    for epoch in range(EPOCHS):
      model.train()
      
      dataset = DynamicGraphDataset(
        topo_gen, Config, 
        max_samples_per_epoch=TOTAL_SAMPLES, 
        rank=rank, 
        world_size=world_size
      )
      
      train_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        pin_memory=True
      )

      total_loss = 0.0
      total_acc = 0.0
      num_batches = 0
      
      if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
      else:
        pbar = train_loader

      for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        edge_logits = model(batch)
        y_true = batch.y
        
        loss = loss_fn(edge_logits, y_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        current_loss = loss.item()
        predicted = (edge_logits > 0.0).float()
        current_acc = (predicted == y_true).float().mean().item()
        
        total_loss += current_loss
        total_acc += current_acc
        num_batches += 1
        
        if rank == 0:
          pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2%}"})

      # 汇总 (Reduce)
      metrics = torch.tensor([total_loss, total_acc, num_batches], device=device)
      dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
      
      global_batches = metrics[2].item()
      if global_batches > 0:
        avg_loss = metrics[0].item() / global_batches
        avg_acc = metrics[1].item() / global_batches
      else:
        avg_loss, avg_acc = 0.0, 0.0

      # Rank 0 负责更新和保存
      if rank == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} [DDP]. Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}, LR: {current_lr:.2e}")
        
        scheduler.step(avg_acc)

        if avg_acc > best_acc:
          best_acc = avg_acc
          torch.save(model.module.state_dict(), SAVE_PATH)
          print(f"💾 Saved Best: {best_acc:.2%}")
      
      # [修复 Barrier 警告 2] 显式指定 device_ids
      dist.barrier(device_ids=[rank])

  finally:
    cleanup()

# === 主程序 ===
if __name__ == "__main__":
  world_size = torch.cuda.device_count()
  if world_size < 1:
    print("❌ 需要 GPU 才能运行此脚本")
  else:
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
