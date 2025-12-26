from .unet import UNet

_model_dict = {
    "UNet_ISIC": UNet,
    "UNet": UNet,
}

def get_model(model_name, **kwargs):
    if model_name not in _model_dict:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(_model_dict.keys())}")
    
    return _model_dict[model_name](**kwargs)