from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
from distillers.dclip import compute_dclip_loss

def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _to_device(x: Any, device: str) -> Any:
    return x.to(device) if (x is not None and hasattr(x, 'to')) else x


def _ce_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target, reduction='none')


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def _teacher_weights_ca_kd(logits_t_list, targets: Optional[torch.Tensor]) -> torch.Tensor:
    K = len(logits_t_list)
    assert K >= 1
    B = logits_t_list[0].shape[0]

    if K == 1:
        return torch.ones(1, B, device=logits_t_list[0].device)

    if targets is not None:
        ce_list = [_ce_per_sample(lg, targets) for lg in logits_t_list]
        L = torch.stack(ce_list, dim=0)
    else:
        ent_list = [_entropy_from_logits(lg) for lg in logits_t_list]
        L = torch.stack(ent_list, dim=0)

    w = (1.0 - F.softmax(L, dim=0)) / (K - 1)
    return w


def _kl_per_sample(student_logits: torch.Tensor,
                   teacher_logits: torch.Tensor,
                   T: float) -> torch.Tensor:
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    p_t = F.softmax(teacher_logits / T, dim=-1)
    kl = F.kl_div(log_p_s, p_t, reduction='none').sum(dim=-1)  # [B]
    return kl * (T * T)


def _solve_ridge(Sd: torch.Tensor, Td: torch.Tensor, ridge: float) -> torch.Tensor:
    Ds = Sd.shape[1]
    StS = Sd.transpose(0, 1) @ Sd
    StT = Sd.transpose(0, 1) @ Td
    eye = torch.eye(Ds, device=Sd.device, dtype=Sd.dtype)
    A = StS + ridge * eye
    try:
        W = torch.linalg.solve(A, StT)
    except RuntimeError:
        W = torch.linalg.pinv(A) @ StT
    return W


def _proj_mse_per_sample(student_feat: torch.Tensor,
                         teacher_feat: torch.Tensor,
                         ridge: float = 1e-3) -> torch.Tensor:
    B = student_feat.shape[0]
    Sd = student_feat.view(B, -1)
    Td = teacher_feat.view(B, -1)

    with torch.no_grad():
        W = _solve_ridge(Sd, Td, ridge)

    T_hat = Sd @ W
    mse = F.mse_loss(T_hat, Td, reduction='none').mean(dim=-1)
    return mse


def _branch_amb_weights(student_logits_dict: Dict[str, torch.Tensor],
                        targets: Optional[torch.Tensor],
                        gamma: float = 3.0,
                        bounds: Tuple[float, float] = (0.5, 2.0)) -> Dict[str, float]:
    names = ['joint', 'image', 'text']
    if targets is None:
        return {n: 1.0 for n in names}

    ce_vals = []
    for n in names:
        ce = F.cross_entropy(student_logits_dict[f'{n}_logits'], targets, reduction='mean')
        ce_vals.append(ce.detach())

    ce_t = torch.stack(ce_vals)
    mean_ce = ce_t.mean().clamp_min(1e-12)
    raw = (ce_t / mean_ce).pow(gamma)
    low, high = bounds
    raw = torch.clamp(raw, min=low, max=high)
    scale = 3.0 / raw.sum()
    w = (raw * scale).tolist()
    return {n: float(wi) for n, wi in zip(names, w)}


def _level_adaptive_weights(
        kd_branch: torch.Tensor,
        feat_branch: torch.Tensor,
        config: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    use_level_adapt = bool(_cfg_get(config, 'use_level_adapt', True))
    gamma_level     = float(_cfg_get(config, 'gamma_level', 2.0))
    level_bounds    = _cfg_get(config, "level_bounds", (0.5, 2.0))

    lam_logits = float(_cfg_get(config, 'lambda_logits', 0.25))
    lam_feat   = float(_cfg_get(config, 'lambda_feat',   0.1))

    if not use_level_adapt:
        # 不自适应：按固定比例权重（归一化）
        total = lam_logits + lam_feat + 1e-12
        w_logit = kd_branch.new_tensor(lam_logits / total)
        w_feat  = kd_branch.new_tensor(lam_feat   / total)
        return w_logit, w_feat

    # ===== 1. 层级不一致性 Difficulty =====
    # 推荐版本：差异 + 本层级难度（更鲁棒）
    D_logit = abs(kd_branch.detach()  - feat_branch.detach()) + kd_branch.detach()
    D_feat  = abs(kd_branch.detach()  - feat_branch.detach()) + feat_branch.detach()

    # ===== 2. 归一化 =====
    D_mean = (D_logit + D_feat) / 2 + 1e-12
    R_logit = (D_logit / D_mean).pow(gamma_level)
    R_feat  = (D_feat  / D_mean).pow(gamma_level)

    # ===== 3. 截断 =====
    low, high = level_bounds
    R_logit = torch.clamp(R_logit, min=low, max=high)
    R_feat  = torch.clamp(R_feat,  min=low, max=high)

    # ===== 4. Softmax 风格归一化 =====
    sum_R = R_logit + R_feat + 1e-12
    w_logit = R_logit / sum_R
    w_feat  = R_feat  / sum_R
    return w_logit, w_feat


def cosine_distillation_loss(student_embeddings, teacher_embeddings):
    """
    Cosine similarity loss: 1 - cos(student, teacher)
    """
    student_norm = F.normalize(student_embeddings, dim=1)
    teacher_norm = F.normalize(teacher_embeddings, dim=1)
    cos_sim = torch.sum(student_norm * teacher_norm, dim=1)  # [B]
    return torch.mean(1.0 - cos_sim)


def compute_contrastive_loss(image_embeddings, text_embeddings, device=None, temperature=0.05):
    """
    InfoNCE contrastive loss between image and text embeddings
    """
    image_embeddings = F.normalize(image_embeddings, dim=1)
    text_embeddings = F.normalize(text_embeddings, dim=1)

    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    batch_size = image_embeddings.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


def compute_amkd_loss(outputs_t1: Dict[str, torch.Tensor],
                      outputs_t2: Dict[str, torch.Tensor],
                      outputs_s: Dict[str, torch.Tensor],
                      targets,
                      device: str = 'cpu',
                      config: Any = None) -> torch.Tensor:

    # ===== 1. 读取配置 =====
    T = float(_cfg_get(config, 'T', 2))
    ridge = float(_cfg_get(config, 'ridge', 5e-4))
    lam_logits = float(_cfg_get(config, 'lambda_logits', 0.24))
    lam_feat = float(_cfg_get(config, 'lambda_feat', 0.1))
    w_branch = _cfg_get(config, 'w_branch', None) or {'joint': 1.0, 'image': 1.0, 'text': 1.0}

    # 额外损失权重
    w_cdist = float(getattr(config, "cosine_loss_weight", 4.0))
    w_contrast = float(getattr(config, "contrastive_loss_weight", 7.5))
    temperature = float(getattr(config, "temperature", 0.06))

    # 标签（仅分类任务会用到）
    if isinstance(config, dict):
        targets = config.get('targets', None)
    else:
        targets = getattr(config, 'targets', None)
    if targets is not None:
        targets = _to_device(targets, device)

    # ===== 2. 在 fill_keys 之前，用原始 outputs_s 判断是否有 logits =====

    def has_logits(d: Dict[str, torch.Tensor], key: str) -> bool:
        return (d is not None) and (key in d) and (d[key] is not None)

    has_joint_logits = has_logits(outputs_s, 'joint_logits')
    has_img_logits   = has_logits(outputs_s, 'image_logits')
    has_txt_logits   = has_logits(outputs_s, 'text_logits')

    is_classification = has_joint_logits or has_img_logits or has_txt_logits
    is_retrieval = not is_classification

    # ===== 3. 搬到 device 并填充缺失 key（为了后续代码统一访问） =====

    def move_pack(d):
        return {k: _to_device(v, device) for k, v in d.items()}

    t1 = move_pack(outputs_t1)
    t2 = move_pack(outputs_t2)
    s  = move_pack(outputs_s)

    def fill_keys(x: dict, device: str):
        zero = torch.tensor([0.], device=device)
        x['joint_feat']  = x.get('joint_feat',  None) if x.get('joint_feat',  None) is not None else zero
        x['image_feat']  = x.get('image_feat',  None) if x.get('image_feat',  None) is not None else zero
        x['text_feat']   = x.get('text_feat',   None) if x.get('text_feat',   None) is not None else zero

        x['joint_logits'] = x.get('joint_logits', None) if x.get('joint_logits', None) is not None else zero
        x['image_logits'] = x.get('image_logits', None) if x.get('image_logits', None) is not None else zero
        x['text_logits']  = x.get('text_logits',  None) if x.get('text_logits',  None) is not None else zero
        return x

    t1 = fill_keys(t1, device)
    t2 = fill_keys(t2, device)
    s  = fill_keys(s,  device)

    branches = ['joint', 'image', 'text']

    # ===== 4. AMB 分支权重（仅分类任务启用） =====
    use_amb = bool(_cfg_get(config, 'use_amb', targets is not None))
    if is_classification and use_amb and (targets is not None):
        amb_gamma  = float(_cfg_get(config, 'amb_gamma', 2.6))
        amb_bounds = _cfg_get(config, 'amb_bounds', (0.8, 1.6))
        w_adapt = _branch_amb_weights(
            {f'{n}_logits': s[f'{n}_logits'] for n in branches},
            targets, gamma=amb_gamma, bounds=amb_bounds
        )
    else:
        w_adapt = {br: 1.0 for br in branches}

    # 用某个张量初始化 total_loss
    total_loss = s['joint_feat'].new_tensor(0.0)

    # ===== 5. 分类损失：只有在 is_classification=True 时计算 =====
    if is_classification:
        for br in branches:
            # KD（logits 蒸馏）
            w_k = _teacher_weights_ca_kd(
                [t1[f'{br}_logits'], t2[f'{br}_logits']],
                targets
            )
            w1, w2 = w_k[0], w_k[1]

            kd1 = _kl_per_sample(s[f'{br}_logits'], t1[f'{br}_logits'], T)
            kd2 = _kl_per_sample(s[f'{br}_logits'], t2[f'{br}_logits'], T)
            kd_branch = (w1 * kd1 + w2 * kd2).mean()

            # 特征蒸馏（Proj-MSE）
            f1 = _proj_mse_per_sample(s[f'{br}_feat'], t1[f'{br}_feat'], ridge=ridge)
            f2 = _proj_mse_per_sample(s[f'{br}_feat'], t2[f'{br}_feat'], ridge=ridge)
            feat_branch = (w1 * f1 + w2 * f2).mean()

            # debug_wts_20251126
            w_logit, w_feat = _level_adaptive_weights(kd_branch, feat_branch, config)
            branch_level_loss = w_logit * kd_branch + w_feat * feat_branch
            

            branch_weight = float(w_branch.get(br, 1.0)) * float(w_adapt.get(br, 1.0))

            total_loss = total_loss + branch_weight * branch_level_loss

    # ===== 6. 检索相关损失：cosine 蒸馏 + InfoNCE（只在 is_retrieval=True 时启用） =====

    if is_retrieval:
        # 教师1蒸馏
        loss_img_t1 = cosine_distillation_loss(s["image_feat"], t1["image_feat"])
        loss_txt_t1 = cosine_distillation_loss(s["text_feat"], t1["text_feat"])
        # 教师2蒸馏
        loss_img_t2 = cosine_distillation_loss(s["image_feat"], t2["image_feat"])
        loss_txt_t2 = cosine_distillation_loss(s["text_feat"], t2["text_feat"])

        # student 自身 image-text 对比学习
        contrastive_loss = compute_contrastive_loss(
            s["image_feat"], s["text_feat"],
            device=device,
            temperature=temperature
        )

        # —— 这是你希望的“检索损失”写法（略作括号修正）——
        retrieval_loss = (
            w_cdist * (loss_img_t1 + loss_txt_t1 + loss_img_t2 + loss_txt_t2)
            + w_contrast * contrastive_loss
        )

        total_loss = total_loss + retrieval_loss

    return total_loss