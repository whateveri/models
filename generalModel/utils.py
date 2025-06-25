
import torch
from pathlib import Path




def save_model(model:torch.nn.Module,
               path:str,
               model_name:str):
    model_dir=Path(path)
    model_dir.mkdir(parents=True,
                    exist_ok=True)

    assert model_name.endswith("pth") or model_name.endswith("pt"), "model name should not ends with 'pt' or 'pth'"
    model_save_path=model_dir / model_name

    print(f'Model already saved in {model_save_path}')

    torch.save(obj=model.state_dict(),f=model_save_path)
