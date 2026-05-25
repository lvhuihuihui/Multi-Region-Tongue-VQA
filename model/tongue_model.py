# tongue_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from .visual_encoder import QwenVLVisionEncoder
from .region_aware_fusion import RegionAwareFusion
from .tongue_gnn import TongueGNN

class MLPProjector(nn.Module):
    """2层 MLP 投影器 + 零初始化（理论：初期不破坏 LLM 原生语义流形）"""
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x): return self.net(x)

class TongueFusionGNNModel(nn.Module):
    def __init__(self, qwen_path, vl_path=None, d_r=512, d_v=768):
        super().__init__()
        
        # 1. 视觉编码器
        self.visual_encoder = QwenVLVisionEncoder(
            model_path=vl_path or "/home/qluai/lvhui/models/Qwen3-VL-4B-Instruct",
            d_out=d_v, use_lora=True
        )
        
        # 2. 多模态融合 & 3. 图结构建模
        self.fusion = RegionAwareFusion(d_in=d_v, d_r=d_r, d_v=d_v)
        self.gnn = TongueGNN(d_r=d_r, d_g=d_v)

        # 4. 语言模型
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token