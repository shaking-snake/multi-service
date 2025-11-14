import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # 导入 DataLoader 和 Dataset
import numpy as np # 导入 numpy
from tqdm import tqdm # 导入进度条库

from .TmpClassifier import RNNClassifier
from ..PreferenceModule import PreferenceModule 
from .FlowGenerator import FlowGenerator, FlowFingerprintDataset, normalize_fingerprint, N_PACKETS, NUM_CLASSES

# 训练参数
EPOCHS = 20
INPUT_DIM = 2 
RNN_LAYERS = 2
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
RNN_HIDDEN_DIM = 128 
TEST_SET_SIZE = BATCH_SIZE * 300   
TRAIN_SET_SIZE = BATCH_SIZE * 500 
PRETRAINED_PATH = "./MS/LSTM/Pretrain/Classify-model.pth"

def evaluate(model, test_loader, criterion, device):
  """在测试集上评估模型的性能"""
  model.eval()  # 设置模型为评估模式 (禁用 Dropout, Batch Norm 等)
  test_loss = 0.0
  test_correct = 0
  test_total = 0
  
  # 禁用梯度计算，节省内存并加速推理
  with torch.no_grad():
    # test_pbar = tqdm(test_loader, desc="[Evaluate]")
    for fingerprints, labels in test_loader:
      fingerprints, labels = fingerprints.to(device), labels.to(device)
      
      logits = model(fingerprints)
      loss = criterion(logits, labels)
      
      # 统计
      test_loss += loss.item() * labels.size(0)
      _, predicted = torch.max(logits.data, 1)
      test_total += labels.size(0)
      test_correct += (predicted == labels).sum().item()
      
      # 更新进度条
      # test_pbar.set_postfix({
      #   'Loss': f"{test_loss / test_total:.4f}",
      #   'Acc': f"{100 * test_correct / test_total:.2f}%"
      # })

  avg_loss = test_loss / test_total
  avg_acc = 100 * test_correct / test_total
  
  return avg_loss, avg_acc

def train_phase_1a():
    
  # --- 0. 环境设置 ---
  # 动态选择设备 (优先使用 GPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"--- 阶段 1.A：预训练 RNN 分类器开始 ---")
  print(f"使用设备: {device}, D_rnn: {RNN_HIDDEN_DIM}, Batch Size: {BATCH_SIZE}")

  # --- 1. 实例化数据 ---
  # 实例化 FlowGenerator
  generator = FlowGenerator(num_packets=N_PACKETS) 
  
  # 实例化动态数据集和 DataLoader
  train_dataset = FlowFingerprintDataset(generator, epoch_size=TRAIN_SET_SIZE)
  train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 

  # 测试数据集 (动态生成，但评估时使用)
  test_dataset = FlowFingerprintDataset(generator, epoch_size=TEST_SET_SIZE)
  test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

  # --- 2. 实例化模型、损失函数和优化器 ---
  # 实例化 RNNClassifier (注意：RNNClassifier 内部包含了 PreferenceModule)
  model = RNNClassifier(
    input_dim=INPUT_DIM,
    rnn_hidden_dim=RNN_HIDDEN_DIM,
    rnn_layers=RNN_LAYERS,
    num_classes=NUM_CLASSES
  ).to(device) # 将整个模型移动到 GPU/CPU
  
  # 损失函数：交叉熵，用于多分类
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
  
  # 将保存标准从最低训练损失改为最高的测试准确率
  best_test_acc = -1.0 

  # --- 3. 训练循环 ---
  for epoch in range(EPOCHS):
    model.train() # 设置为训练模式
    train_loss = 0.0
    train_correct = 0
    train_samples = 0
    
    # --- A. 训练阶段 ---
    # train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for fingerprints, labels in train_loader:
      fingerprints, labels = fingerprints.to(device), labels.to(device)
      
      optimizer.zero_grad() 
      logits = model(fingerprints) 
      loss = criterion(logits, labels) 
      loss.backward() 
      optimizer.step() 
      
      train_loss += loss.item() * labels.size(0)
      _, predicted = torch.max(logits.data, 1)
      train_samples += labels.size(0)
      train_correct += (predicted == labels).sum().item()
      
      # train_pbar.set_postfix({
      #   'Loss': f"{train_loss / train_samples:.4f}",
      #   'Acc': f"{100 * train_correct / train_samples:.2f}%"
      # })
        
    avg_train_loss = train_loss / train_samples
    avg_train_acc = 100 * train_correct / train_samples

    # --- B. 评估阶段 ---
    avg_test_loss, avg_test_acc = evaluate(model, test_loader, criterion, device)

    # --- C. Epoch 总结 ---
    print(f"Epoch {epoch+1} 总结: Train Acc: {avg_train_acc:.2f}% | Test Acc: {avg_test_acc:.2f}%")

    # --- 4. 核心产出：保存最佳权重 (基于测试准确率) ---
    if avg_test_acc > best_test_acc:
      best_test_acc = avg_test_acc
      
      # 仅保存 LSTM Body (preference_module) 的权重
      torch.save(
        model.state_dict(), 
        PRETRAINED_PATH
      )
  print(f"最佳测试准确率: {best_test_acc:.2f}%. 保存 [LSTM Body] 权重到 {PRETRAINED_PATH}")

# --- 运行代码 ---
if __name__ == '__main__':
  train_phase_1a()