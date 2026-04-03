import torch
from torch.utils.data import random_split, DataLoader


from datasets.mmimdb import MMIMDbCLIPDataset
from datasets.vqav2 import VQAv2Dataset
from datasets.flickr30k import Flickr30kRetrievalDataset
from datasets.mscoco import COCORetrievalDataset


def load_dataset(dataset, batch_size=32, val_split=0.1, test_split=0.1, image_size=224,
                 num_workers=8, pin_memory=True):
    """
    返回:
        train_loader, val_loader, test_loader, number_category
    说明:
        - 分类任务: 返回 number_category>0（用于走分类分支）
        - 检索任务: 返回 number_category=None（用于走检索分支）
    """

    # 1) 避免 if 连写导致“贯穿执行”，改用 if/elif/elif/elif
    if dataset == 'mmimdb':
        full_dataset = MMIMDbCLIPDataset(image_size=image_size)
        total_len = len(full_dataset)
        number_category = 23

        # 随机划分
        val_len = int(val_split * total_len)
        test_len = int(test_split * total_len)
        train_len = total_len - val_len - test_len
        generator = torch.Generator().manual_seed(0)
        train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    elif dataset == 'vqav2':
        full_dataset = VQAv2Dataset(image_size=image_size)
        total_len = len(full_dataset)
        number_category = 3129 

        # 随机划分
        val_len = int(val_split * total_len)
        test_len = int(test_split * total_len)
        train_len = total_len - val_len - test_len
        generator = torch.Generator().manual_seed(0)
        train_set, val_set, test_set = random_split(full_dataset, [train_len, val_len, test_len], generator=generator)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    elif dataset == 'flickr-30k':
        # 检索任务：显式传入 image_size（数据类若支持）
        train_set = Flickr30kRetrievalDataset(split="train", image_size=image_size)
        val_set   = Flickr30kRetrievalDataset(split="val",   image_size=image_size)
        test_set  = Flickr30kRetrievalDataset(split="test",  image_size=image_size)
        number_category = None  # 检索分支要求是假值

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    elif dataset == 'ms-coco':
        try:
            # 如果数据类支持集合/列表 split，优先走这条
            train_set = COCORetrievalDataset(split={"train", "restval"}, image_size=image_size)
        except Exception:
            # 兼容：数据类只接受字符串 split，则手工拼接
            from torch.utils.data import ConcatDataset
            train_set = ConcatDataset([
                COCORetrievalDataset(split="train",   image_size=image_size),
                COCORetrievalDataset(split="restval", image_size=image_size),
            ])

        val_set   = COCORetrievalDataset(split="val",  image_size=image_size)
        test_set  = COCORetrievalDataset(split="test", image_size=image_size)
        number_category = None  # 检索分支

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
        val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
        test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_loader, val_loader, test_loader, number_category
