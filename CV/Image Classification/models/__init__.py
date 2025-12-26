from .resnet import ResNet50, ResNet101, ResNet152
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from .vggnet import VGGNet11, VGGNet13, VGGNet16, VGGNet19

_model_dict = {
    "ResNet50": ResNet50, "ResNet101": ResNet101, "ResNet152": ResNet152,
    "DenseNet121": DenseNet121, "DenseNet169": DenseNet169, "DenseNet201": DenseNet201, "DenseNet264": DenseNet264,
    "VGGNet11": VGGNet11, "VGGNet13": VGGNet13, "VGGNet16": VGGNet16, "VGGNet19": VGGNet19,
}

def get_model(model_name, **kwargs):
    if model_name not in _model_dict:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(_model_dict.keys())}")
    
    return _model_dict[model_name](**kwargs)