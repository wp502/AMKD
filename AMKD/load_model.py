from models.clip import CLIPWrapper, CLIPMultiModalClassifier
from models.resnet_bert import ResNetBertWrapper, ResNetBertMultiModalClassifier
from models.vit_bert import ViTBertWrapper, ViTBertMultiModalClassifier


def parse_model_id(model_id: str):
    """
    解析组合字段：
    - 'clip-ViT-B-32' → ('clip', 'ViT-B-32')
    - 'clip-RN50' → ('clip', 'RN50')
    - 'resnet-bert' → ('resnet+bert', None)
    - 'vit-bert' → ('vit+bert', None)
    """
    if model_id.startswith('clip-'):
        model_name = model_id[len('clip-'):]
        return 'clip', model_name

    elif model_id == 'resnet-bert':
        return 'resnet-bert', None

    elif model_id == 'vit-bert':
        return 'vit-bert', None

    else:
        raise ValueError(f"Invalid model_id: {model_id}. Supported formats: 'clip-<model>', 'resnet-bert', 'vit-bert'")


def build_model(model_id: str, num_category, project_dim):
   
    model_type, model_name = parse_model_id(model_id)

    if model_type == "clip":
        clip_model = CLIPWrapper(model_name=model_name, pretrained_source='openai', project_dim=project_dim)
        return clip_model if num_category is None else CLIPMultiModalClassifier(clip_model, num_category)

    elif model_type == "resnet-bert":
        wrapper = ResNetBertWrapper(project_dim=project_dim)
        return wrapper if num_category is None else ResNetBertMultiModalClassifier(wrapper, num_category)

    elif model_type == "vit-bert":
        wrapper = ViTBertWrapper(project_dim=project_dim)
        return wrapper if num_category is None else ViTBertMultiModalClassifier(wrapper, num_category)

    else:
        raise ValueError(f"Unknown model_type: {model_type}")