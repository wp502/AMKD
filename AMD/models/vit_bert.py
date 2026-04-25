import torch
import torch.nn as nn
import timm
from transformers import BertModel, BertTokenizer


class ViTBertWrapper(nn.Module):
    def __init__(self, freeze=False, device="cpu", project_dim=None):
        super().__init__()
        self.device = device
        self.D = project_dim

        self.visual_backbone = timm.create_model("vit_base_patch16_224", pretrained=False)
        ckpt_path = "raw_models/vit/vit-b-16.pth"
        state_dict = torch.load(ckpt_path, map_location=device)
        self.visual_backbone.load_state_dict(state_dict)
        self.visual_proj = nn.Linear(self.visual_backbone.num_features, self.D)

        bert_path = "raw_models/bert/bert-base-uncased"
        self.textual_backbone = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.text_proj = nn.Linear(self.textual_backbone.config.hidden_size, self.D)

        if freeze:
            for p in self.visual_backbone.parameters():
                p.requires_grad = False
            for p in self.textual_backbone.parameters():
                p.requires_grad = False

    def encode_image(self, image):
        feat = self.visual_backbone.forward_features(image) # [B, 197, 768]
        mean_feat = feat.mean(dim=1)  # [B, 768]
        return self.visual_proj(mean_feat)  # [B, D]

    def encode_text(self, text_input):
        feat = self.textual_backbone(**text_input).pooler_output  # [B, 768]
        return self.text_proj(feat)  # [B, D]

    def forward(self, image, text_input):
        return {
            "image_feat": self.encode_image(image),
            "text_feat": self.encode_text(text_input)
        }

    def get_tokenizer(self):
        return self.tokenizer

    def get_output_dim(self):
        return self.D


class ViTBertMultiModalClassifier(nn.Module):
    def __init__(self, backbone: ViTBertWrapper, num_category: int):
        super().__init__()
        self.backbone = backbone
        self.D = backbone.get_output_dim()
        self.num_category = num_category

        self.joint_proj = nn.Linear(2 * self.D, self.D)
        self.joint_classifier = nn.Linear(self.D, num_category)
        self.image_classifier = nn.Linear(self.D, num_category)
        self.text_classifier = nn.Linear(self.D, num_category)

    def forward(self, image, text_input):
        image_feat = self.backbone.encode_image(image)
        text_feat = self.backbone.encode_text(text_input)
        joint_feat = self.joint_proj(torch.cat([image_feat, text_feat], dim=-1))
        return {
            "joint_logits": self.joint_classifier(joint_feat),
            "image_logits": self.image_classifier(image_feat),
            "text_logits": self.text_classifier(text_feat),
            "joint_feat": joint_feat,
            "image_feat": image_feat,
            "text_feat": text_feat
        }

    def get_tokenizer(self):
        return self.backbone.get_tokenizer()