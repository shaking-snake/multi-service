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
import re

from .DijkstraGnn import GNNPretrainModel, GLOBAL_STATS, DynamicGraphDataset
from ...Env.NetworkGenerator import TopologyGenerator

# === [新增] 核心武器：Focal Loss ===
class FocalLoss(nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, logits=True, reduce=True):
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

    if self.reduce:
      return torch.mean(F_loss)
    else:
      return F_loss

if __name__ == "__main__":
  try:
    torch.multiprocessing.set_start_method('spawn', force=True)
  except RuntimeError: pass
  try:
    torch.multiprocessing.set_sharing_strategy('file_system')
  except RuntimeError: pass

  print("🚀 开始阶段 1B: GNN 主体预训练 (终极冲刺 - Focal Loss + 热重启)...")

  # --- 1. 超参数配置 ---
  EPOCHS = 3000          # 保持 400，配合热重启需要更多轮次
  GNN_DIM = 256
  NUM_LAYERS = 6
  BATCH_SIZE = 128       
  SAMPLES_PER_EPOCH = 6400
  
  # [调整] 配合热重启，初始 LR 可以稍微给高一点点，让它有能力跳出坑
  LEARNING_RATE = 2e-7
  
  NODE_FEAT_DIM = 5
  EDGE_FEAT_DIM = 2

  # [断点续训] 建议设置为你最新的最好模型，例如 Epoch 290+ 的
  RESUME_PATH = "./MS/GNN/Pretrain/pretrained_model.pth"

  if torch.cuda.is_available():
    torch.cuda.init()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # --- 2. 初始化组件 ---
  topo_gen = TopologyGenerator(num_nodes_range=(20, 30), m_ba=2)
  model = GNNPretrainModel(NODE_FEAT_DIM, GNN_DIM, EDGE_FEAT_DIM, NUM_LAYERS)
  
  swa_model = AveragedModel(model) # 创建 SWA 模型影子
  swa_start = 300 # 从第 300 轮开始收集 SWA 权重
  
  
  start_epoch = 0
  if RESUME_PATH is not None and os.path.exists(RESUME_PATH):
    print(f"🔄 正在从 {RESUME_PATH} 加载检查点...")
    state_dict = torch.load(RESUME_PATH, map_location='cpu')
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    print("✅ 模型权重加载成功！")
    start_epoch = 300

  model = model.to(device)
  if torch.cuda.device_count() > 1:
    print(f"✨ 启用 {torch.cuda.device_count()} 张 GPU 进行 PyG DataParallel 加速")
    model = DataParallel(model)
      
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # [微调] 加一点点 weight_decay 防止过拟合
  swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
  # [核心升级 1] 使用余弦退火热重启调度器
  # T_0=50: 首次重启周期为 50 Epoch
  # T_mult=1: 之后每次重启周期保持 50 Epoch (你可以设为 2 让周期变长)
  # eta_min=1e-6: 学习率最低降到 1e-6
  from torch.optim import lr_scheduler
  scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-10
  )

  # [核心升级 2] 使用 Focal Loss 替代 BCE
  # alpha=0.85: 强烈增加正样本（最短路径边）的权重，因为它们太少了
  # gamma=2.0: 标准的困难样本聚焦参数
  loss_fn = FocalLoss(alpha=0.85, gamma=2.0, logits=True)

  best_acc = 0.9975
  # --- 3. 训练循环 ---
  for epoch in range(start_epoch, EPOCHS):
    model.train()
    dataset = DynamicGraphDataset(topo_gen, GLOBAL_STATS, max_samples=SAMPLES_PER_EPOCH)
    train_loader = DataListLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

    for batch_data_list in pbar:
      optimizer.zero_grad()
      edge_logits = model(batch_data_list)
      y_true = torch.cat([data.y for data in batch_data_list]).to(device)
      
      loss = loss_fn(edge_logits, y_true) # 使用 Focal Loss
      
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()

      current_loss = loss.item()
      total_loss += current_loss
      predicted = (edge_logits > 0.0).float()
      current_acc = (predicted == y_true).float().mean().item()
      total_acc += current_acc
      num_batches += 1
      pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2%}"})

    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    
    # 注意：CosineAnnealingWarmRestarts 需要在每次 step() 后更新，或者每 epoch 更新
    # 这里我们在 epoch 结束时更新。注意它不需要传入验证集 loss。

    scheduler.step(epoch + 1 / EPOCHS) # 原来的调度器

    print(f"Epoch {epoch+1} 完成. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2%}, LR: {current_lr:.2e}")

    if avg_acc > best_acc:
      model_to_save = model.module if isinstance(model, DataParallel) else model
      torch.save(model_to_save.state_dict(), RESUME_PATH)
      best_acc = avg_acc

  print("✅ 阶段 1B 预训练终极完成！")