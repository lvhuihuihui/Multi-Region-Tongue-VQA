# model/text_encoder.py
import os
import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class RegionTextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()

        # ✅ 优先使用本地已下载的模型路径
        local_clip_path = "/home/qluai/lvhui/structured_GAN_QwenV10/code/model/clip-vit-base-patch32"
        
        # 检查本地路径是否存在且有效
        if os.path.isdir(local_clip_path) and os.path.exists(os.path.join(local_clip_path, "config.json")):
            print(f"✅ 加载本地 CLIP 文本编码器：{local_clip_path}")
            cache_dir = None
            model_path = local_clip_path
            local_files_only = True
        else:
            print(f"⚠️ 本地路径未找到，尝试从缓存/网络加载：{model_name}")
            cache_dir = os.getenv("HF_CACHE_DIR", "/home/qluai/.cache/huggingface/hub")
            model_path = model_name
            local_files_only = False

        # 加载 Tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # 加载 Text Encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        
        # 冻结参数
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, region_texts):
        device = next(self.text_encoder.parameters()).device
        
        inputs = self.tokenizer(
            region_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        ).to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
        
        return outputs.last_hidden_state[:, -1, :]  # (N, 512)
