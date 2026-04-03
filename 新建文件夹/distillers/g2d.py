import torch
import torch.nn.functional as F

def compute_g2d_loss(outputs_t1, outputs_t2, outputs_s, config=None):
    """
    双多模态教师分别对学生模型进行模态蒸馏（G2D 改进版）：
    - 每个教师都与学生模型在 joint/image/text 三种输出上对齐（logits + features）
    """

    # 提取超参数（提供默认值以防未设置）
    temp = getattr(config, "temperature", 2.0) if config else 2.0
    logit_weight = getattr(config, "logit_loss_weight", 1.0)
    feature_weight = getattr(config, "feature_loss_weight", 1.0)

    # KL 蒸馏函数（logits 监督）
    def kl_divergence(student_logits, teacher_logits, temp):
        t_soft = F.softmax(teacher_logits / temp, dim=1)
        s_log_soft = F.log_softmax(student_logits / temp, dim=1)
        return F.kl_div(s_log_soft, t_soft, reduction='batchmean') * (temp ** 2)

    # MSE 特征监督函数
    def mse_loss(student_feat, teacher_feat):
        return F.mse_loss(student_feat, teacher_feat)

    # === 教师模型 1 对学生蒸馏 ===
    loss_t1 = (
        kl_divergence(outputs_s["joint_logits"], outputs_t1["joint_logits"], temp) +
        kl_divergence(outputs_s["image_logits"], outputs_t1["image_logits"], temp) +
        kl_divergence(outputs_s["text_logits"], outputs_t1["text_logits"], temp) +
        mse_loss(outputs_s["joint_feat"], outputs_t1["joint_feat"]) +
        mse_loss(outputs_s["image_feat"], outputs_t1["image_feat"]) +
        mse_loss(outputs_s["text_feat"], outputs_t1["text_feat"])
    )

    # === 教师模型 2 对学生蒸馏 ===
    loss_t2 = (
        kl_divergence(outputs_s["joint_logits"], outputs_t2["joint_logits"], temp) +
        kl_divergence(outputs_s["image_logits"], outputs_t2["image_logits"], temp) +
        kl_divergence(outputs_s["text_logits"], outputs_t2["text_logits"], temp) +
        mse_loss(outputs_s["joint_feat"], outputs_t2["joint_feat"]) +
        mse_loss(outputs_s["image_feat"], outputs_t2["image_feat"]) +
        mse_loss(outputs_s["text_feat"], outputs_t2["text_feat"])
    )

    # 总损失：两个教师的蒸馏损失之和，按权重加权
    total_loss = logit_weight * (loss_t1 + loss_t2) + feature_weight * (loss_t1 + loss_t2)

    return total_loss
