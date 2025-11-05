import torch
import torch.nn as nn
import torch.optim as optim

from .DijkstraGnn import get_pyg_data_from_nx, generate_expert_label, GNNPretrainModel, GLOBAL_STATS, DynamicGraphDataset, DataLoader
from ...Env.NetworkGenerator import TopologyGenerator

if __name__ == "__main__":
  print("🚀 开始阶段 1B: GNN 主体预训练 (Mini-batch DataLoader 模式)...")

  # --- 1. 超参数配置 ---
  EPOCHS = 100          # 总轮数可以适当减少，因为现在每轮看的图多了
  GNN_DIM = 128         # 建议增加宽度
  NUM_LAYERS = 6        # [重要] 建议增加深度以覆盖网络直径
  BATCH_SIZE = 64       # [关键] 真正的小批量大小
  LEARNING_RATE = 1e-3  # Batch变大后，学习率通常可以稍微调大一点
  SAMPLES_PER_EPOCH = 6400 # 每个 epoch 总共看多少张图 (6400 / 64 = 100 steps)
  
  NODE_FEAT_DIM = 3     # 假设你采用了我之前建议的 BFS Hop 特征，如果是旧的则为 3
  EDGE_FEAT_DIM = 2

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # --- 2. 初始化组件 ---
  topo_gen = TopologyGenerator(num_nodes_range=(20, 30), m_ba=2)                           # topo generator
  model = GNNPretrainModel(NODE_FEAT_DIM, GNN_DIM, EDGE_FEAT_DIM, NUM_LAYERS).to(device)   # gnn model
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)                             # optimizer 

  # [重要] 固定 pos_weight 以防止震荡。
  # 根据经验，20-30节点的图，负边大约是正边的 15-25 倍。
  # 我们取一个保守值 15.0，让模型稍微多预测一点正样本，保证 Recall。
  POS_WEIGHT_FIXED = torch.tensor([15.0]).to(device)
  loss_fn = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT_FIXED)

  # FiLM 中和参数
  GAMMA_NEUTRAL = torch.ones((NUM_LAYERS, GNN_DIM), dtype=torch.float).to(device)
  BETA_NEUTRAL = torch.zeros((NUM_LAYERS, GNN_DIM), dtype=torch.float).to(device)

  # --- 3. 训练循环 ---
  for epoch in range(EPOCHS):
    model.train()
  
    # [关键] 每个 epoch 重新创建 Dataset 和 DataLoader
    # 这是为了让新的 epoch 能生成新的随机图，保持数据的无限多样性
    dataset = DynamicGraphDataset(topo_gen, GLOBAL_STATS, max_samples_per_epoch=SAMPLES_PER_EPOCH)
    # num_workers > 0 可以多进程生成图，加速训练，但可能需要处理一些多进程共享种子的细节
    # 这里先用 num_workers=0 (主进程生成) 保证简单稳定
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0) 

    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    # 使用 tqdm 包装 loader 以显示进度条
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    
    for batch_data in pbar:
      # batch_data 是一个 "巨图"，包含了 BATCH_SIZE (e.g., 64) 个小图
      # 它的 .edge_index, .x, .edge_attr 都是自动拼接好的
      batch_data = batch_data.to(device)
      
      optimizer.zero_grad()
      
      # 前向传播：模型像处理一个大图一样处理这个 batch
      edge_logits = model(batch_data, manual_gamma=GAMMA_NEUTRAL, manual_beta=BETA_NEUTRAL)
      
      # 计算损失：batch_data.y 包含了这 64 个图的所有边的标签
      loss = loss_fn(edge_logits, batch_data.y)
      
      loss.backward()
      
      # [可选] 梯度裁剪，防止爆炸
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      
      optimizer.step()

      # 统计指标
      current_loss = loss.item()
      total_loss += current_loss
      
      # 简单准确率
      predicted = (edge_logits > 0.0).float()
      current_acc = (predicted == batch_data.y).float().mean().item()
      total_acc += current_acc
      num_batches += 1
      
      # 更新进度条显示
      pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.2%}"})

    # Epoch 结束总结
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    print(f"Epoch {epoch+1} 完成. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.2%}")

  # 6. 保存预训练好的 GNN 主体
  # 注意：保存的是整个模型，在阶段 2 加载时需要选择性加载
  print("✅ 阶段 1B 完成。保存 GNN 主体权重...")
  # 我们只保存 GNN 主体（卷积层、归一化层）和节点嵌入层的权重
  # 丢弃 self.edge_output_head
  gnn_body_weights = {k: v for k, v in model.state_dict().items() if 'edge_output_head' not in k}
  torch.save(gnn_body_weights, 'pretrained-model-with-posWeight.pth')