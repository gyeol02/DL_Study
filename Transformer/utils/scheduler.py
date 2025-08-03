import torch

def get_transformer_scheduler(optimizer, d_model, warmup_steps=4000):
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)