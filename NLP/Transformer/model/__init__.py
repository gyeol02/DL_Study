from .transformer import Transformer

_model_dict = {
    "Transformer": Transformer,
}

def get_model(model_name, **kwargs):
    if model_name not in _model_dict:
        raise ValueError(f"Model '{model_name}' not found.")
    return _model_dict[model_name](**kwargs)