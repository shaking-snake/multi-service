import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import DataParallel
from torch_geometric.loader import DataListLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from tqdm import tqdm
import torch.multiprocessing
import os
import math
import numpy as np
import random
from torch.utils.data import IterableDataset

# === 导入自定义模块 ===
from MS.GNN.FiLMGnn import FiLMGnn
from MS.Env.NetworkGenerator import TopologyGenerator, get_pyg_data_from_nx, DEFAULT_CONFIG
from Pretrain.gnn.TmpDataGenerator import generate_expert_label, DynamicGraphDataset

# === 1. 训练配置 (与 NetworkGenerator 保持一致) ===
NETX_CONFIG = DEFAULT_CONFIG

# === 2. 动态数据集 (适配 Config) ===
def generate_single_sample(topo_gen, config):
  """不断生成随机拓扑样本"""
  while True:
    try:
      # 1. 生成拓扑
      G_nx = topo_gen.generate_topology()
      # 2. 随机选点
      S, D = topo_gen.select_source_destination()
      # 3. 提取特征 (8维节点, 4维边)
      data, G_with_attrs = get_pyg_data_from_nx(G_nx, S, D, config)
      # 4. 计算标签 (Dijkstra 最短路)
      y = generate_expert_label(G_with_attrs, S, D, data.edge_index)
      
      if y is not None:
        data.y = y
        return data
    except Exception:
      continue # 生成失败重试

class DynamicGraphDataset(IterableDataset):
    def __init__(self, topo_gen, config, max_samples):
        self.topo_gen = topo_gen
        self.config = config
        self.max_samples = max_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 确保多进程下随机数种子不同
            worker_seed = torch.initial_seed() % (2**32 - 1) + worker_info.id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        
        # 计算每个 Worker 需要生成的样本数
        if worker_info is None:
            iter_end = self.max_samples
        else:
            per_worker = int(math.ceil(self.max_samples / float(worker_info.num_workers)))
            iter_end = min(worker_info.id * per_worker + per_worker, self.max_samples)
            
        for _ in range(iter_end):
            yield generate_single_sample(self.topo_gen, self.config)

# === 3. 损失函数 (Focal Loss) ===
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=2.0, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # 强力增加正样本(最短路边)权重
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

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# === 主程序 ===
if __name__ == "__main__":
    # 多进程设置
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    print("🚀 开始 Phase 1.B: GNN 全特征预训练 (Node=8, Edge=4)...")

    # --- 超参数 ---
    EPOCHS = 1000          
    BATCH_SIZE = 128       
    SAMPLES_PER_EPOCH = 6400
    LEARNING_RATE = 5e-4
    
    # [关键] 这里的维度必须与 NetworkGenerator 生成的一致
    NODE_FEAT_DIM = 8   # [Deg, Src, Dst, DistS, DistD, Betw, Clus, PR]
    EDGE_FEAT_DIM = 4   # [Delay, BW, Loss, Util]
    GNN_DIM = 256
    NUM_LAYERS = 6

    SAVE_PATH = "./trained_model/gnn_pretrained_model.pth"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 初始化 ---
    topo_gen = TopologyGenerator(Config) # 传入配置类
    
    model = GNNPretrainModel(
        node_feat_dim=NODE_FEAT_DIM, 
        gnn_dim=GNN_DIM, 
        edge_feat_dim=EDGE_FEAT_DIM, 
        num_layers=NUM_LAYERS
    )
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"✨ 启用 {torch.cuda.device_count()} 张 GPU 加速")
        model = DataParallel(model)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    loss_fn = FocalLoss(alpha=0.85, gamma=2.0)

    best_acc = 0.0

    # --- 训练循环 ---
    for epoch in range(EPOCHS):
        model.train()
        
        # 动态生成数据集
        dataset = DynamicGraphDataset(topo_gen, Config, max_samples=SAMPLES_PER_EPOCH)
        train_loader = DataListLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for batch_data_list in pbar:
            optimizer.zero_grad()
            
            # 前向传播 (Dijkstra 模式: gamma=1, beta=0)
            edge_logits = model(batch_data_list)
            
            # 拼接标签
            y_true = torch.cat([data.y for data in batch_data_list]).to(device)
            
            # 计算损失
            loss = loss_fn(edge_logits, y_true)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 统计
            current_loss = loss.item()
            total_loss += current_loss
            
            # 准确率 (Logits > 0 即为选中)
            predicted = (edge_logits > 0.0).float()
            current_acc = (predicted == y_true).float().mean().item()
            
            total_acc += current_acc
            num_batches += 1
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2%}"})

        avg_loss = total_loss / num_batches
        avg_acc  = total_acc / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新学习率
        scheduler.step(avg_acc)

        print(f"Epoch {epoch+1} 结束. Loss: {avg_loss:.4f}, Acc: {avg_acc:.2%}, LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_acc > best_acc:
            best_acc = avg_acc
            model_to_save = model.module if isinstance(model, DataParallel) else model
            torch.save(model_to_save.state_dict(), SAVE_PATH)
            print(f"💾 新的最佳模型已保存! Best Acc: {best_acc:.2%}")

    print("✅ GNN 全特征预训练完成！")