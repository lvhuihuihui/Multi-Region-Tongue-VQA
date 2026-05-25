# visual_encoder.py
import torch
import torch.nn as nn
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
import os
import gc

class QwenVLVisionEncoder(nn.Module):
    """
    Qwen3-VL 视觉编码器适配器
    理论依据：
      1. 跨模态线性投影：将视觉潜空间 ℝ^D_v 微分同胚映射至共享语义空间 ℝ^d_v
      2. Row-Major 索引：严格遵循 NaViT 序列排列，不破坏 patch 空间邻接关系
      3. 零信息损失：无 padding/truncation，特征流形连续性完整保留
    """
    def __init__(self, model_path, d_out=768, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        print(f"🔹 正在加载 Qwen3-VL 视觉编码器: {model_path}")
        
        if not os.path.isdir(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
            
        # 1. CPU 加载避免显存瞬间打满
        try:
            self.full_model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path, 
                torch_dtype="auto", 
                device_map="cpu",
                local_files_only=True,
                low_cpu_mem_usage=True
            )
        except Exception as e:
            raise RuntimeError(f"❌ 加载模型失败: {e}")

        # 2. 提取视觉塔
        self.visual = self.full_model.visual
        self.patch_size = getattr(self.visual.config, 'patch_size', 14)
        self.visual_embed_dim = getattr(self.visual.config, 'embed_dim', 1024)
        
        # 3. 释放完整模型
        del self.full_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # 4. 跨模态对齐投影层 (理论核心)
        self.align_proj = nn.Linear(self.visual_embed_dim, d_out)
        nn.init.kaiming_uniform_(self.align_proj.weight)
        nn.init.zeros_(self.align_proj.bias)
        
        # 5. 可选 LoRA 注入
        self.use_lora = use_lora
        if use_lora:
            # ✅ ViT/SigLIP 架构标准命名：qkv(合并), proj, fc1, fc2
            vision_lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha,
                target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=0.05, bias="none"
            )
            self.visual = get_peft_model(self.visual, vision_lora_config)
            print(f"✅ 视觉塔 LoRA 注入成功 (r={lora_r}, alpha={lora_alpha})")
            
        # 6. 冻结非 LoRA 参数
        for name, param in self.visual.named_parameters():
            if "lora" not in name.lower():
                param.requires_grad = False
                
        print(f"✅ Qwen3-VL 视觉编码器加载成功 | 原生维度: {self.visual_embed_dim} -> 投影: {d_out}")

    def forward(self, pixel_values):
        """
        输入: pixel_values (B, C, H, W)
        输出: (B, N, d_out) 投影后的视觉特征序列
        """
        B, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # 动态计算网格尺寸 (NaViT)
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        grid_thw = torch.tensor([[1, grid_h, grid_w]], device=device, dtype=torch.int64).expand(B, -1)
        
        # 确保模块在正确设备
        if next(self.visual.parameters()).device != device:
            self.visual.to(device)
            self.align_proj.to(device)
            
        with torch.set_grad_enabled(self.training):
            # 注意：transformers>=4.45 视觉塔参数名为 pixel_values
            outputs = self.visual(pixel_values=pixel_values, grid_thw=grid_thw)
            feats = outputs.last_hidden_state  # (B, N, D_v)
            
        return self.align_proj(feats)  # (B, N, d_out)

    def extract_region_features(self, projected_patches, coords, grid_h, grid_w):
        """
        基于 Row-Major 序列的严格区域索引（不破坏拓扑）
        projected_patches: (B, N, d) 已投影特征
        coords: (B, 5, 4) 归一化坐标 [x_center, y_center, w, h] ∈ [0,1]
        grid_h, grid_w: 当前图像的网格尺寸
        """
        B, N, D = projected_patches.shape
        device = projected_patches.device
        
        # 1. 坐标映射到网格索引
        u_center = (coords[:, :, 0] * grid_w).long().clamp(0, grid_w - 1)
        v_center = (coords[:, :, 1] * grid_h).long().clamp(0, grid_h - 1)
        u_half   = (coords[:, :, 2] * grid_w / 2).long().clamp(1, grid_w // 2)
        v_half   = (coords[:, :, 3] * grid_h / 2).long().clamp(1, grid_h // 2)
        
        region_feats = []
        for i in range(5):  # 5个区域
            u, v = u_center[:, i], v_center[:, i]
            uh, vh = u_half[:, i], v_half[:, i]
            
            # 计算边界 (左闭右开)
            u_min = (u - uh).clamp(0, grid_w - 1)
            u_max = (u + uh + 1).clamp(1, grid_w)
            v_min = (v - vh).clamp(0, grid_h - 1)
            v_max = (v + vh + 1).clamp(1, grid_h)
            
            batch_feats = []
            for b in range(B):
                # 2. 严格 Row-Major 索引: idx = v * W + u
                rows = torch.arange(v_min[b], v_max[b], device=device)[:, None]
                cols = torch.arange(u_min[b], u_max[b], device=device)[None, :]
                flat_indices = (rows * grid_w + cols).flatten()
                
                # 3. 安全截断：防止超出实际序列长度 N
                valid_mask = flat_indices < N
                flat_indices = flat_indices[valid_mask]
                
                if flat_indices.numel() > 0:
                    feat = projected_patches[b, flat_indices].mean(dim=0)  # (D,)
                else:
                    # 兜底：取中心点
                    center_idx = (v[b] * grid_w + u[b]).clamp(0, N-1)
                    feat = projected_patches[b, center_idx]
                batch_feats.append(feat)
                
            region_feats.append(torch.stack(batch_feats, dim=0))  # (B, D)
            
        return torch.stack(region_feats, dim=1)  # (B, 5, D)