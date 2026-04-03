import os
import torch

def load_pretrained_teacher(model, model_name, dataset_name, project_dim):
    """从 HRAD/raw_models/teachers/train_new/{dataset_name} 路径中加载预训练权重"""
    model_tag = model_name.replace("/", "-")  # 避免 huggingface 名称中的 /
    path = os.path.join("raw_models", "teachers", "train", dataset_name, f"{model_tag}_{dataset_name}_best.pth")
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"✅ Loaded pretrained weights for [{model_tag}] on [{dataset_name}] from: {path}")
    else:
        print(f"⚠️  No pretrained weights found at: {path} (training from scratch)")
