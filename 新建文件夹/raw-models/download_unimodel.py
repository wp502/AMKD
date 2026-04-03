import os
import torch
import timm
import open_clip
from transformers import BertModel, BertTokenizer
import torchvision.models as models

save_dir_1 = "./vit/"
save_dir_2 = "./bert/"
save_dir_3 = "./resnet/"

'''
# ==== 下载 ViT-B-16 (CLIP) ====
print("Downloading ViT-B-16 (pure vision model from timm)")
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
torch.save(vit_model.state_dict(), os.path.join(save_dir_1, "vit-b-16.pth"))
print("Saved ViT-B-16 to raw_models/vit-b-16.pth")
'''


# ==== 下载 BERT-Base ====
print("Downloading BERT-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert.save_pretrained(os.path.join(save_dir_2, "bert-base-uncased"))
tokenizer.save_pretrained(os.path.join(save_dir_2, "bert-base-uncased"))
print(f"Saved BERT-base to {save_dir_2}/bert-base-uncased")

'''
# ==== 下载 ResNet-18 ====
print("Downloading ResNet18")
resnet18 = models.resnet18(pretrained=True)
torch.save(resnet18.state_dict(), os.path.join(save_dir_3, "resnet18.pth"))
print(f"Saved ResNet18 to {save_dir_3}")
'''