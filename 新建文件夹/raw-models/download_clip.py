# download_clip_models.py
import os
import torch
import open_clip

# 教师和学生模型列表
clip_models = [
    "RN101", 
]

# 保存路径
save_dir = "./clip/"
os.makedirs(save_dir, exist_ok=True)

for model_name in clip_models:
    print(f"Downloading: {model_name}")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), path)
    print(f"Saved {model_name} to {path}")
