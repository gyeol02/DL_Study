import os
import json
import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    
    print(f"Setting Global Seed to {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    # CUDA 관련
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("   ✅ CUDA Deterministic Mode Enabled")
        
    # MPS 관련 - manual_seed만으로도 대부분의 초기화는 고정됨
    if torch.backends.mps.is_available():
        print("   ✅ MPS Seed Set (Note: Some MPS ops might not be fully deterministic yet)")

def create_experiment_folders(base_path, model_name):
   
    model_root = os.path.join(base_path, model_name)
    os.makedirs(model_root, exist_ok=True)

    setting_number = 1
    while True:
        folder_name = f"setting_#{setting_number}"
        exp_dir = os.path.join(model_root, folder_name)
        
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            log_dir = os.path.join(exp_dir, "logs")
            ckpt_dir = os.path.join(exp_dir, "models")
            os.makedirs(log_dir)
            os.makedirs(ckpt_dir)
            
            print(f"📂 Created Experiment Folder: {exp_dir}")
            return log_dir, ckpt_dir, exp_dir
            
        setting_number += 1

def save_config(config, save_path):
    
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = config

    with open(os.path.join(save_path, "config_saved.json"), "w") as f:
        json.dump(config_dict, f, indent=4)