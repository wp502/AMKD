import os
import torch
import torch.nn as nn
import open_clip

class CLIPWrapper(nn.Module):
    def __init__(self, 
                 model_name='ViT-B-32', 
                 pretrained_source='openai', 
                 freeze=False, 
                 project_dim=None,
                 device='cpu'):
      
        super().__init__()
        self.model_name = model_name
        self.device = device
        custom_ckpt_path = f"raw_models/clip/{model_name}.pth"

   
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained_source if custom_ckpt_path is None else None,
            device=device
        )

        if custom_ckpt_path:
            print(f"[CLIPWrapper] Loading weights from {custom_ckpt_path}")
            state_dict = torch.load(custom_ckpt_path, map_location=device)
            self.model.load_state_dict(state_dict)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

       
        self.output_dim = self.model.visual.output_dim
        if project_dim is not None:
            self.image_proj = nn.Linear(self.output_dim, project_dim)
            self.text_proj = nn.Linear(self.output_dim, project_dim)
            self.output_dim = project_dim  
        else:
            self.image_proj = self.text_proj = nn.Identity()


    def forward(self, image, text):
       
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return {
            "image_feat": image_features,
            "text_feat": text_features
        }

    def encode_image(self, image):
        feat = self.model.encode_image(image)
        return self.image_proj(feat)

    def encode_text(self, text):
        feat = self.model.encode_text(text)
        return self.text_proj(feat)

    def get_output_dim(self):
        return self.output_dim
        
    def get_preprocess_train(self):
        return self.preprocess_train
    
    def get_preprocess_val(self):
        return self.preprocess_val


class CLIPMultiModalClassifier(nn.Module):
    def __init__(self, clip_model, num_category):
        super().__init__()
        self.clip = clip_model
        self.D = clip_model.get_output_dim()
        self.num_category = num_category

       
        self.joint_proj = nn.Linear(2 * self.D, self.D)
        self.joint_classifier = nn.Linear(self.D, num_category)

        self.image_classifier = nn.Linear(self.D, num_category)
        self.text_classifier = nn.Linear(self.D, num_category)

    def forward(self, image, text_tokens):
        image_feat = self.clip.encode_image(image)   # [B, D]
        text_feat = self.clip.encode_text(text_tokens)  # [B, D]

        joint_feat = torch.cat([image_feat, text_feat], dim=-1)  # [B, 2D]
        joint_feat = self.joint_proj(joint_feat)  # [B, D]
        joint_logits = self.joint_classifier(joint_feat)  # [B, num_category]

        
        img_logits = self.image_classifier(image_feat)
        txt_logits = self.text_classifier(text_feat)

        return {
            "joint_logits": joint_logits,
            "image_logits": img_logits,
            "text_logits": txt_logits,
            "joint_feat": joint_feat,
            "image_feat": image_feat,
            "text_feat": text_feat
        }

    def encode_image(self, image):
        return self.clip.encode_image(image)

    def encode_text(self, text):
        return self.clip.encode_text(text)

    def get_preprocess_train(self):
        return self.clip.preprocess_train()

    def get_preprocess_val(self):
        return self.clip.preprocess_val()