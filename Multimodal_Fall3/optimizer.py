import torch
from timm.scheduler import (
    StepLRScheduler,
    MultiStepLRScheduler,
    CosineLRScheduler,
)

def build_optimizer(model,optim_type,optim_params,scheduler_type,scheduler_params):
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),**optim_params)
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(),**optim_params)
    elif optim_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),**optim_params)
    elif optim_type == "rms":
        optimizer = torch.optim.RMSprop(model.parameters(), **optim_params)
    else:
        raise NotImplementedError(f"Optimizer type [{optim_type}] is not implemented. Available type [ sgd, adam, adamw ].")
    
    if scheduler_type == None:
        lr_scheduler = None
    elif scheduler_type == "step":
        lr_scheduler = StepLRScheduler(optimizer,**scheduler_params)
    elif scheduler_type == "multistep":
        lr_scheduler = MultiStepLRScheduler(optimizer,**scheduler_params)
    elif scheduler_type == "cosine":
        lr_scheduler = CosineLRScheduler(optimizer,**scheduler_params)
    else:
        raise NotImplementedError(f"Optimizer type [{optim_type}] is not implemented. Available type [ step, multistep, cosine ].")
    
    return optimizer, lr_scheduler    