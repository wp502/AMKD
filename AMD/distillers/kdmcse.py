import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)   # non-linear activation
        return x


class ArcSimilarity(nn.Module):
    def __init__(self, temp, margin=0.05, eps=1e-6):
        super().__init__()
        self.temp = float(temp)
        self.margin = float(margin)
        self.eps = float(eps)
        self.cos = nn.CosineSimilarity(dim=-1, eps=eps)

    def _safe_acos(self, x: torch.Tensor) -> torch.Tensor:
     
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = torch.clamp(x, -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(x)

    def calculate_arccos1(self, cos_sim, labels=None, slabels=None):
        theta = self._safe_acos(cos_sim)
        if labels is None:
            labels = torch.arange(cos_sim.size(0), device=cos_sim.device).long()
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes).to(theta.device)
        selected_labels = torch.where(
            torch.gt(theta, math.pi - self.margin),
            torch.zeros_like(one_hot_labels),
            one_hot_labels
        )
        final_theta = torch.where(
            selected_labels.bool(),
            theta + self.margin,
            theta
        )
        out = torch.cos(final_theta)
        return torch.nan_to_num(out, nan=0.0)

    def calculate_arccos2(self, cos_sim, labels=None, slabels=None):
        theta = self._safe_acos(cos_sim)
        if labels is None:
            labels = torch.arange(cos_sim.size(0), device=cos_sim.device).long()
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes).to(theta.device)

       
        selected_labels = torch.where(
            torch.gt(theta, self.margin),
            torch.ones_like(one_hot_labels),
            one_hot_labels
        ) * torch.abs(one_hot_labels - 1)

        if slabels is None:
            final_theta = torch.where(selected_labels.bool(), theta - self.margin, theta)
        else:
            
            final_theta = torch.where(selected_labels.bool(), theta - (1 - slabels) * self.margin, theta)

        out = torch.cos(final_theta)
        return torch.nan_to_num(out, nan=0.0)

    def forward(self, x, y, slabels=None):
        sim = self.cos(x, y)
        sim = self.calculate_arccos2(sim, slabels=slabels)
        return sim / self.temp


class ClipVisnModel(nn.Module):
    def __init__(self, feature_dim, proj_dim):
        super().__init__()
        self.vmlp = MLPLayer(feature_dim, proj_dim)   # visual features -> grounding space
        self.tmlp = MLPLayer(feature_dim, proj_dim)   # textual features -> grounding space
        self.register_buffer("logit_scale", torch.tensor(np.log(1 / 0.05), dtype=torch.float32))
        self.loss_fct = nn.CrossEntropyLoss()

    def logit(self, image_features, text_features):
        device = image_features.device
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        ground_truth = torch.arange(logits_per_image.size(0), device=device)
        total_loss = (self.loss_fct(logits_per_image, ground_truth) +
                      self.loss_fct(logits_per_text, ground_truth)) / 2
        return total_loss

    def forward(self, visn_feat, text_feat):
        visn_feat = F.normalize(self.vmlp(visn_feat), p=2, dim=-1, eps=1e-6)
        text_feat = F.normalize(self.tmlp(text_feat), p=2, dim=-1, eps=1e-6)
        return visn_feat, text_feat, None  # self.logit(visn_feat, text_feat)


class MCSE(nn.Module):
    def __init__(self, use_threshold: bool = False, threshold_cos: float = 0.7):
        super().__init__()
        self.grounding = MLPLayer(512, 256)
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.Tanh()  # 与 BERT pooler 一致
        )
        self.visn_model = ClipVisnModel(512, 256)
        self.sim = ArcSimilarity(temp=0.05, margin=0.15)
        self.sim_vl = ArcSimilarity(temp=0.05, margin=0.15)
        self.loss_fct = nn.CrossEntropyLoss()

        self.using_threshold = bool(use_threshold)
        self.threshold_cos = float(threshold_cos)
        if self.using_threshold:
            print(f"USING THRESHOLD with cos > {self.threshold_cos:.2f}")

    def forward(self, outputs_s):
        num_sent = 2
        lang_pooled_output = outputs_s['text_feat'].unsqueeze(1).expand(-1, num_sent, -1)  # [bs, 2, 512]
        lang_projection = self.projection(lang_pooled_output)  # [bs, 2, 512]
        return lang_pooled_output, lang_projection

    def compute_loss(self, outputs_t1, outputs_t2, outputs_s, cal_inter=False):
        l_pool, l_proj = self.forward(outputs_s)
        # Separate representation
        z1, z2 = l_proj[:, 0], l_proj[:, 1]  # (bs, hidden)

        # Intra-sentence contrast
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)
        labels = torch.arange(cos_sim.size(0), device=cos_sim.device).long()
        loss = self.loss_fct(cos_sim, labels)  # unsup: bs-1 negatives

        if not cal_inter:
            return loss

        # ---------- Inter-modality ----------
        img_feat = outputs_t1['image_feat']     # [bs, hidden_dim]
        text_feat = outputs_t1['text_feat']     # [bs, hidden_dim]

       
        vis_feats_t = F.normalize(img_feat, p=2, dim=1, eps=1e-6)         # [bs, hidden_dim]
        caption_feats_t = F.normalize(text_feat, p=2, dim=1, eps=1e-6)    # [bs, hidden_dim]

       
        vv_scores_ = vis_feats_t @ vis_feats_t.t()                        # [bs, bs], cos
        cc_scores_ = caption_feats_t @ caption_feats_t.t()                # [bs, bs], cos


        if self.using_threshold:
            batch_size = img_feat.size(0)
            device = img_feat.device
            eye_mask = ~torch.eye(batch_size, dtype=bool, device=device)

            vc_cos = vis_feats_t @ caption_feats_t.t()                    # [bs, bs], cos
            cv_cos = caption_feats_t @ vis_feats_t.t()                    # [bs, bs], cos

            vc_labels = torch.where(torch.logical_and(vc_cos > self.threshold_cos, eye_mask),
                                    torch.ones_like(vc_cos), torch.zeros_like(vc_cos)) * -99999.99
            cv_labels = torch.where(torch.logical_and(cv_cos > self.threshold_cos, eye_mask),
                                    torch.ones_like(cv_cos), torch.zeros_like(cv_cos)) * -99999.99
        
        l2v_proj = F.normalize(self.grounding(l_pool), p=2, dim=-1, eps=1e-6)  # [bs, 2, proj_dim]
        p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]                                 # [bs, proj_dim]

       
        v_1, t_1, _ = self.visn_model(outputs_t1["image_feat"], outputs_t1["text_feat"])  # [bs, proj_dim]
        cos_sim_p0_v = self.sim_vl(p1.unsqueeze(1), v_1.unsqueeze(0), slabels=vv_scores_)  # (bs, bs)
        cos_sim_p1_v = self.sim_vl(p2.unsqueeze(1), v_1.unsqueeze(0), slabels=vv_scores_)
        cos_sim_p0_t = self.sim_vl(p1.unsqueeze(1), t_1.unsqueeze(0), slabels=cc_scores_)  # (bs, bs)
        cos_sim_p1_t = self.sim_vl(p2.unsqueeze(1), t_1.unsqueeze(0), slabels=cc_scores_)
        if self.using_threshold:
            cos_sim_p0_v = cos_sim_p0_v + cv_labels
            cos_sim_p1_v = cos_sim_p1_v + cv_labels
            cos_sim_p0_t = cos_sim_p0_t + vc_labels
            cos_sim_p1_t = cos_sim_p1_t + vc_labels
        inter_loss1 = (self.loss_fct(cos_sim_p0_v, labels) + self.loss_fct(cos_sim_p1_v, labels)) / 2
        inter_loss2 = (self.loss_fct(cos_sim_p0_t, labels) + self.loss_fct(cos_sim_p1_t, labels)) / 2

    
        v_2, t_2, _ = self.visn_model(outputs_t2["image_feat"], outputs_t2["text_feat"])  # [bs, proj_dim]
        cos_sim_p0_v_2 = self.sim_vl(p1.unsqueeze(1), v_2.unsqueeze(0), slabels=vv_scores_)  # (bs, bs)
        cos_sim_p1_v_2 = self.sim_vl(p2.unsqueeze(1), v_2.unsqueeze(0), slabels=vv_scores_)
        cos_sim_p0_t_2 = self.sim_vl(p1.unsqueeze(1), t_2.unsqueeze(0), slabels=cc_scores_)  # (bs, bs)
        cos_sim_p1_t_2 = self.sim_vl(p2.unsqueeze(1), t_2.unsqueeze(0), slabels=cc_scores_)
        if self.using_threshold:
            cos_sim_p0_v_2 = cos_sim_p0_v_2 + cv_labels
            cos_sim_p1_v_2 = cos_sim_p1_v_2 + cv_labels
            cos_sim_p0_t_2 = cos_sim_p0_t_2 + vc_labels
            cos_sim_p1_t_2 = cos_sim_p1_t_2 + vc_labels
        inter_loss3 = (self.loss_fct(cos_sim_p0_v_2, labels) + self.loss_fct(cos_sim_p1_v_2, labels)) / 2
        inter_loss4 = (self.loss_fct(cos_sim_p0_t_2, labels) + self.loss_fct(cos_sim_p1_t_2, labels)) / 2

        inter_loss = inter_loss1 + inter_loss2 + inter_loss3 + inter_loss4
        return loss, inter_loss


def compute_kdmcse_loss(outputs_t1, outputs_t2, outputs_s, model: MCSE):
    intra_loss, inter_loss = model.compute_loss(outputs_t1, outputs_t2, outputs_s, cal_inter=True)
    loss = intra_loss + 0.01 * inter_loss
    return loss
