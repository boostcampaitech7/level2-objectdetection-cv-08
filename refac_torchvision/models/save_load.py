import os
import torch

def save_model(model, save_path):
    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), save_path)

def load_model(model, load_path, device):
    model.load_state_dict(torch.load(load_path, map_location=device))
    return model
