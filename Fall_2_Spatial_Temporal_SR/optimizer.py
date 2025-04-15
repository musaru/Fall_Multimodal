import torch
from timm.scheduler import (
    StepLRScheduler,
    MultiStepLRScheduler,
    CosineLRScheduler,
)

def build_optimizer(model,config):
    optim_type = config.OPTIM.TYPE
    
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.OPTIM.LR, momentum=config.OPTIM.MOMENTUM, weight_decay=config.OPTIM.WEIGHT_DECAY)
        
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.OPTIM.LR, betas=config.OPTIM.BETAS, eps=config.OPTIM.EPS, weight_decay=config.OPTIM.WEIGHT_DECAY)
        
    elif optim_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.OPTIM.LR, betas=config.OPTIM.BETAS, eps=config.OPTIM.EPS, weight_decay=config.OPTIM.WEIGHT_DECAY)

    elif optim_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.OPTIM.LR)
        
    else:
        raise RuntimeError(f"Optimizer type [{optim_type}] is not implemented.")

    scheduler_type = config.LR_SCHEDULER.TYPE
    
    if scheduler_type == None:
        lr_scheduler = None
    elif scheduler_type == "cosine":
        lr_scheduler = CosineLRScheduler(optimizer, t_initial=config.LR_SCHEDULER.T_INITIAL, lr_min=config.LR_SCHEDULER.LR_MIN, t_in_epochs=config.LR_SCHEDULER.T_IN_EPOCHS, warmup_t=config.LR_SCHEDULER.WARMUP_T, warmup_lr_init=config.LR_SCHEDULER.WARMUP_LR_INIT)
    else:
        raise RuntimeError(f"LR Scheduler type [{scheduler_type}] is not implemented.")
    
    return optimizer, lr_scheduler    

def _build_optimizer(model,optim_type,optim_params,scheduler_type,scheduler_params):
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),**optim_params)
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(),**optim_params)
    elif optim_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(),**optim_params)
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