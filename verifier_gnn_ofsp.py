import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
from MS.GNN.Pretrain.DijkstraGnn import GNNPretrainModel, get_pyg_data_from_nx, generate_expert_label, GLOBAL_STATS
from MS.Env.NetworkGenerator import TopologyGenerator

# ================= 配置 =================
# 必须与 Training.py 中的参数完全一致
MODEL_PATH = "./MS/GNN/Pretrain/pretrained_model.pth" # 你的模型路径
NODE_FEAT_DIM = 5
GNN_DIM = 256
EDGE_FEAT_DIM = 2
NUM_LAYERS = 6

def test_gnn_performance():
  # 1. 准备环境
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"正在加载模型: {MODEL_PATH} (Device: {device})")
  
  model = GNNPretrainModel(NODE_FEAT_DIM, GNN_DIM, EDGE_FEAT_DIM, NUM_LAYERS).to(device)
  
  try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    # 兼容 DataParallel 保存的 'module.' 前缀
    if list(state_dict.keys())[0].startswith('module.'):
      state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print("[ms] 模型加载成功！")
  except FileNotFoundError:
    print(f"❌ 错误：找不到模型文件 {MODEL_PATH}")
    return
  except Exception as e:
    print(f"❌ 模型加载出错: {e}")
    return

  model.eval()
  
  # 2. 初始化拓扑生成器
  topo_gen = TopologyGenerator(num_nodes_range=(15, 25), m_ba=2)
  
  print("="*60)
  print("[ms] 开始测试 GNN 的最短路模仿能力 (共 10000 个随机案例)...")
  
  perfect = 0
  good = 0
  bad = 0
  pbar = tqdm(range(10000))
  for i in pbar:
    # --- 生成数据 ---
    G = topo_gen.generate_topology()
    try:
      s, d = topo_gen.select_source_destination()
    except:
      continue # 跳过连通性问题
        
    # 转换为 PyG 数据
    data, G_with_attrs = get_pyg_data_from_nx(G, s, d, GLOBAL_STATS)
    data = data.to(device)
    
    # 获取真值 (Dijkstra)
    # y_true 是一个 (Num_Edges, ) 的 0/1 向量
    y_true = generate_expert_label(G_with_attrs, s, d, data.edge_index).to(device)
    if y_true is None: continue
    
    # --- 模型推理 ---
    with torch.no_grad():
      # Phase 1B 模式：不传 gamma/beta，让模型使用默认的“中性”参数
      edge_logits = model(data)
      # 使用 Sigmoid 转概率，大于 0.5 (即 logits > 0) 判为路径边
      preds = (edge_logits > 0.0).float()
        
    # --- 指标计算 ---
    # 1. 基础准确率 (所有边的分类正确率)
    correct = (preds == y_true).sum().item()
    total_edges = y_true.size(0)
    acc = correct / total_edges
    
    # 2. 关键指标：Precision & Recall (针对“路径边”即 Label=1)
    # 我们最关心的是：GNN 找出的边，是不是真的是最短路上的边？
    true_positives = ((preds == 1) & (y_true == 1)).sum().item()
    actual_positives = (y_true == 1).sum().item()    # 真实最短路有多少条边
    predicted_positives = (preds == 1).sum().item()  # GNN 认为有多少条边
    
    recall = true_positives / actual_positives if actual_positives > 0 else 0.0
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    
    # print(f"[ms] 节点数: {len(G.nodes())}, 边数: {len(G.edges())}, Src: {s} -> Dst: {d}")
    # print(f"[ms] 真实最短路长度: {actual_positives} 跳")
    # print(f"[ms] GNN 预测路径长度: {predicted_positives} 跳")
    # print(f"[ms] Accuracy (整体): {acc:.2%} (大部分是0，这个高是正常的)")
    # print(f"[ms] Recall (召回率):   {recall:.2%}  <-- GNN 找到了多少条正确的路？")
    # print(f"[ms] Precision (精确率): {precision:.2%} <-- GNN 找的边里有多少是对的？")
    
    if recall == 1.0 and precision == 1.0:
      # print("[ms] 完美复刻！GNN 找到了完全一致的 Dijkstra 路径。")
      perfect = perfect+1
    elif recall > 0.5:
      # print("[ms] 还可以。GNN 找到了大部分路径，可能有轻微偏差。")
      good = good+1
    else:
      # print("[ms] 失败。GNN 迷路了。")
      bad = bad+1
    # print("\n")

  print(f"[ms] perfect: {perfect} good: {good} bad: {bad}")

if __name__ == "__main__":
  test_gnn_performance()