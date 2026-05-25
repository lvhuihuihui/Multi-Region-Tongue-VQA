import torch
import torch.nn as nn
import math
import warnings
from .text_encoder import RegionTextEncoder

class RegionAwareFusion(nn.Module):
    def __init__(self, d_in=768, d_r=512, d_v=768, d_k=64):
        super().__init__()
        self.d_k = d_k
        self.text_encoder = RegionTextEncoder()
        
        # ✅ 新增：适配 Qwen3-VL 输出维度的输入投影
        self.input_proj = nn.Linear(d_in, d_v)
        
        self.WQ = nn.Linear(d_r, d_k)
        self.WK = nn.Linear(d_v, d_k)
        self.WV = nn.Linear(d_r, d_v)
        self.WQ2 = nn.Linear(d_v, d_k)
        self.WK2 = nn.Linear(d_r, d_k)
        self.WV2 = nn.Linear(d_v, d_r)
        
        self.mhsa = nn.MultiheadAttention(embed_dim=d_v, num_heads=8, batch_first=True, dropout=0.05)
        self.global_token = nn.Parameter(torch.randn(1, 1, d_v))
        self.eps = 1e-6
        self.clip_val = 80.0

    def forward(self, patch_features, region_texts):
        if torch.isnan(patch_features).any() or torch.isinf(patch_features).any():
            patch_features = torch.nan_to_num(patch_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        B, N, d_v = patch_features.shape
        device = patch_features.device
        
        # ✅ 对齐输入维度
        patch_features = self.input_proj(patch_features)
        
        r_flat = self.text_encoder(region_texts)
        r = r_flat.view(B, 5, -1)
        r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=-1.0)

        Q, K, V = self.WQ(r), self.WK(patch_features), self.WV(r)
        S = torch.bmm(Q, K.transpose(1, 2)) / (math.sqrt(self.d_k) + self.eps)
        A = torch.softmax(torch.clamp(S, -self.clip_val, self.clip_val), dim=-1)
        v_enhanced = patch_features + torch.bmm(A.transpose(1, 2), V)
        
        v_attended, _ = self.mhsa(v_enhanced, v_enhanced, v_enhanced)
        
        global_query = self.global_token.expand(B, -1, -1)
        attn_scores = torch.clamp(torch.bmm(global_query, v_attended.transpose(1, 2)) / (math.sqrt(d_v) + self.eps), -self.clip_val, self.clip_val)
        r_global = torch.bmm(torch.softmax(attn_scores, dim=-1), v_attended)

        Q2, K2, V2 = self.WQ2(patch_features), self.WK2(r), self.WV2(patch_features)
        S2 = torch.clamp(torch.bmm(Q2, K2.transpose(1, 2)) / (math.sqrt(self.d_k) + self.eps), -self.clip_val, self.clip_val)
        r_updated = r + torch.bmm(torch.softmax(S2, dim=-1).transpose(1, 2), V2)
        
        return torch.nan_to_num(r_updated, nan=0.0), torch.nan_to_num(r_global, nan=0.0)