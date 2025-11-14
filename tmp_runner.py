# 文件名: tmp_runner.py
# (确保此文件在项目根目录)

import os
import sys
import networkx as nx
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.log import setLogLevel
import numpy as np
from mininet.node import OVSKernelSwitch, OVSController 
from mininet.cli import CLI
from mininet.link import TCLink
from functools import partial
# --- 导入你的环境文件 ---
from MS.Env.NetworkGenerator import TopologyGenerator
from MS.Env.FlowGenerator import FlowGenerator, FlowType
from MS.Env.MininetController import get_flow_fingerprint, measure_path_qos
import torch
from MS.LSTM.Pretrain.TmpClassifier import RNNClassifier #
# ========================================

class GraphTopo(Topo):
  def __init__(self, blueprint_g: nx.Graph, **opts):
    Topo.__init__(self, **opts)
    
    for node_id in blueprint_g.nodes():
      self.addSwitch(f's{node_id}', protocols='OpenFlow13')
      self.addHost(f'h{node_id}')
      self.addLink(f'h{node_id}', f's{node_id}', bw=100, delay='0.1ms')

    for u, v, data in blueprint_g.edges(data=True):
      bw = data.get('bandwidth', 100)
      delay = f"{data.get('delay', 1)}ms"
      self.addLink(f's{u}', f's{v}', bw=bw, delay=delay)

def run_simulation():
  """主运行函数"""
  net = None # 【新】确保 net 在 try/finally 外部定义
  
  try:
    # --- 步骤 1: 生成拓扑 (不变) ---
    print("--- 步骤 1: 生成拓扑 [NetworkGenerator] ---")
    topo_gen = TopologyGenerator(num_nodes_range=(5, 8))
    g = topo_gen.generate_topology()
    print(f"拓扑生成完毕: {len(g.nodes())} 个节点, {len(g.edges())} 条边。")

    # --- 步骤 2: 启动 Mininet ---
    print("\n--- 步骤 2: 启动 Mininet ---")
    RemoteCtrl = partial(RemoteController, ip='127.0.0.1', port=6633)

    net = Mininet(
      topo=GraphTopo(g),
      switch=OVSKernelSwitch,
      link=TCLink,
      controller=RemoteCtrl
    )

    net.start()

    # CLI(net)
    # --- 步骤 2.B: 测试 Mininet 连通性 (pingAll) ---
    print("\n--- 步骤 2.B: 测试 Mininet 连通性 (pingAll) ---")
    packet_loss_percentage = net.pingAll()
    
    if packet_loss_percentage == 100:
      print(f"--- 警告: 出现 {packet_loss_percentage:.1f}% 丢包 ---")
      # raise Exception("Mininet 连通性测试失败!")
    else:
      print(f"--- 连通性测试成功 ({packet_loss_percentage}% 丢包) ---")

    print("\n--- 步骤 3: 生成流配置 [FlowGenerator] ---")
    flow_gen = FlowGenerator()
    flow_type, flow_profile = flow_gen.get_random_flow()
    # 【关键】保存真实标签
    true_class_name = flow_type.name 
    print(f"已选择流类型 (真实标签): {true_class_name}")

    # --- 步骤 4.A (保持不变) ---
    print("\n--- 步骤 4.A: 测试流指纹生成 [MininetController] ---")
    s_id, d_id = topo_gen.select_source_destination()
    S_host = net.get(f'h{s_id}')
    D_host = net.get(f'h{d_id}')
    
    fingerprint_matrix = get_flow_fingerprint(S_host, D_host, flow_profile)
    
    if fingerprint_matrix is None:
      print("\n--- 指纹生成失败! ---")
      raise Exception("无法生成指纹, 终止测试。")

    print("\n--- 指纹生成成功! ---")
    print(f"指纹矩阵形状: {fingerprint_matrix.shape}")
    print(fingerprint_matrix)

  except Exception as e:
      print(f"\n--- 仿真出错 ---")
      print(e)
      import traceback
      traceback.print_exc()
      
  finally:
      # 确保 Mininet 总是被停止
      if net:
        print("\n--- 步骤 5: 停止 Mininet ---")
        net.stop()
      
      # 确保 Mininet 被彻底清理
      print("INFO: 运行 mn -c 以防万一...")
      os.system('sudo mn -c')


if __name__ == '__main__':
    # setLogLevel('info')
    
    if os.getuid() != 0:
        print("错误：此脚本必须以 root 权限 (sudo) 运行。")
    else:
        run_simulation()