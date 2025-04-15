import os
import numpy as np
import random
import datetime
from termcolor import colored
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time
import yaml
import argparse
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import (
classification_report,
accuracy_score,
)

from dataloader import (
    build_urfall_dataloader,
    build_imvia_dataloader,
    build_hurup_dataloader,
    build_fukinect_dataloader,
)
from logger import create_logger
from optimizer import build_optimizer

from models.musa_model import Model, adjGraph, Ablation

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cal_top_k_accuracy(output, target, top_k=[1]):
    accuracies = []
    with torch.no_grad():
        max_k = max(top_k)
        # Get the top max_k predictions
        _, pred = output.topk(max_k, dim=1)
        pred = pred.t()
        
        if target.dim() != 1:
            _, target = target.topk(1,dim=1)
        for k in top_k:
            # Check if targets are in the top k predictions
            correct = pred[:k].eq(target.view(1, -1).expand_as(pred[:k]))
            
            # Calculate accuracy
            correct_k = correct.reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(1 / target.size(0)).item()
            accuracies.append(accuracy)
    
    return accuracies

def cal_remaining_time(seconds):
    m = seconds//60
    s = seconds%60
    return f"{m:.0f}:{s:02.0f}"

def visualize_weights_and_gradients(model,writer,epoch):
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         writer.add_histogram(f"{name}.weight", param.data, epoch)
    #         if param.grad is not None:
    #             writer.add_histogram(f"{name}.grad", param.grad, epoch)

    for order,(name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            writer.add_scalar(f"Gradient_Norm/{order}.{name}", grad_norm, epoch)
    
def train(model,dataloader,optimizer,loss_fn,device="cpu",logger=None,writer=None,logging_timing=10,epoch=None,scaler=torch.cuda.amp.GradScaler(),max_norm=10,top_k=[1]):
    accum_iter = 1
    
    model.train()
    running_loss = []
    running_accu = []
    _time_hist =[]
    
    for i,(data,label) in enumerate(dataloader):
        _start_time = time.time()
        
        data = data.to(device)
        label = label.to(device)

        # label_onehot = torch.nn.functional.one_hot(label, num_classes=model.num_classes)
        # label_onehot = label_onehot.float()
        # data, label_onehot = JDA(data, label_onehot,model.in_channels)
        
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            pred = model(data)
            loss = loss_fn(pred,label)
            # loss = loss_fn(pred,label_onehot)
        
        # loss.backward()
        scaler.scale(loss).backward() if scaler != None else loss.backward()
        
        if i%accum_iter==0 or i+1 == len(dataloader):
            # optimizer.step()
            # Unscales the gradients of optimizer's assigned params in-place
            if scaler != None : scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            if scaler != None : torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) 
        
            scaler.step(optimizer) if scaler != None else optimizer.step()
            visualize_weights_and_gradients(model,writer,epoch)
            if scaler != None : scaler.update()
            model.zero_grad()
        
        running_loss += [loss.item()]
        running_accu += [cal_top_k_accuracy(pred,label,top_k=top_k)]
        
        _end_time = time.time()
        _time_hist.append(_end_time - _start_time)
        _remaining_sec = (np.mean(_time_hist))*(len(dataloader)-(i+1))

        
        if i % logging_timing == 0:
            logger.info(f"[{colored('TRAIN','light_green')} Epoch {epoch} : {i}/{len(dataloader)} ({cal_remaining_time(_remaining_sec)})]\t lr : {optimizer.param_groups[0]['lr']:.5f}\tLoss : {running_loss[-1]:.5f}\tAccuracy (Top-1) : {running_accu[-1][0]:.4f}")
                
    logger.info(f"{colored('TRAIN','light_green')} EPOCH SUMMARY: LOSS : {np.mean(running_loss)}\t ACC : {np.mean(running_accu,axis=0)}")
    if writer:
        writer.add_scalar('Loss/train', np.mean(running_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(running_accu,axis=0)[0], epoch)
    
@torch.no_grad()
def valid(model,dataloader,loss_fn,device,logger=None,writer=None,logging_timing=10,epoch=None,top_k=[1]):
    model.eval()
    
    prediction = []
    labels = []
    running_loss = []
    running_accu = []
    _time_hist =[]
    for i,(data,label) in enumerate(dataloader):
        _start_time = time.time()
        data = data.to(device)
        label = label.to(device)
        if label.dim() == 1 :
            label = torch.nn.functional.one_hot(label, num_classes=model.num_classes).float()
        
        if dataloader.dataset.sample_strategy=="k_copies":
            num_copies = dataloader.dataset.num_copies
            all_output = []

            stride = data.size()[2] // num_copies
            for j in range(num_copies):
                X_slice = data[:, :, j * stride: (j + 1) * stride]
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            pred = torch.mean(all_output, dim=1)
            
        else:
            pred = model(data)
        
        loss = loss_fn(pred,label)

        model.zero_grad()
        
        running_loss += [loss.item()]
        running_accu += [cal_top_k_accuracy(pred,label,top_k=top_k)]

        _end_time = time.time()
        _time_hist.append(_end_time - _start_time)
        _remaining_sec = (np.mean(_time_hist))*(len(dataloader)-(i+1))
        
        if i % logging_timing == 0:
            logger.info(f"[{colored('VALID','light_yellow')} Epoch {epoch} : {i}/{len(dataloader)} ({cal_remaining_time(_remaining_sec)})]\tAccuracy (Top-1) : {running_accu[-1][0]:.4}")

        prediction += [pred]
        labels += [label]
        
        # prediction += torch.argmax(pred, dim=1).to('cpu').detach().numpy().copy().tolist()
        # labels += label.to('cpu').detach().numpy().copy().tolist() if label.dim()==1 else torch.argmax(label,dim=1).to('cpu').detach().numpy().copy().tolist()
    prediction = torch.cat(prediction)
    labels = torch.cat(labels)
            
    logger.info(f"{colored('VALID','light_yellow')} EPOCH SUMMARY: LOSS : {np.mean(running_loss)}\tACC : {cal_top_k_accuracy(prediction,labels,top_k=top_k)}")
    if writer:
        writer.add_scalar('Loss/val', np.mean(running_loss), epoch)
        writer.add_scalar('Accuracy/val', cal_top_k_accuracy(prediction,labels,top_k=top_k)[0], epoch)
    
    return cal_top_k_accuracy(prediction,labels,top_k=top_k)

@torch.no_grad()
def test(model,dataloader,loss_fn,device,logger=None,logging_timing=10,epoch=None,top_k=[1]):
    model.eval()

    prediction = []
    labels = []
    running_loss = []
    running_accu = []
    for i,(data,label) in enumerate(dataloader):
        data = data.to(device)
        label = label.to(device)
        if label.dim() == 1 :
            label = torch.nn.functional.one_hot(label, num_classes=model.num_classes).float()

        if dataloader.dataset.sample_strategy=="k_copies":
            num_copies = dataloader.dataset.num_copies
            all_output = []

            stride = data.size()[2] // num_copies
            for j in range(num_copies):
                X_slice = data[:, :, j * stride: (j + 1) * stride]
                output = model(X_slice)
                all_output.append(output)

            all_output = torch.stack(all_output, dim=1)
            pred = torch.mean(all_output, dim=1)
            
        else:
            pred = model(data)
            
        loss = loss_fn(pred,label)

        model.zero_grad()
        
        running_loss += [loss.item()]
        running_accu += [cal_top_k_accuracy(pred,label,top_k=top_k)]
        
        if i % logging_timing == 0:
            logger.info(f"[{colored('TEST','red')} : {i}/{len(dataloader)}]\tAccuracy (Top-1) : {running_accu[-1][0]:.4}")

        prediction += [pred]
        labels += [label]
        
    prediction = torch.cat(prediction)
    labels = torch.cat(labels)
    
    logger.info(f"{colored('TEST','red')} EPOCH SUMMARY: LOSS : {np.mean(running_loss)}\t ACC : {cal_top_k_accuracy(prediction,labels,top_k=top_k)}")
    
    
    prediction = torch.argmax(prediction, dim=1).to('cpu').detach().numpy().copy()
    labels = labels.to('cpu').detach().numpy().copy() if labels.dim()==1 else torch.argmax(labels,dim=1).to('cpu').detach().numpy().copy()
    logger.info(f"[[[[[Classification Report]]]]]\n{classification_report(labels,prediction,digits=5)}")
    
    
    
    
    
def run(config=None):
    SEED = 42
    fix_seed(SEED)

    output_dir_name = "log_"+datetime.datetime.now().isoformat()
    OUTPUT_DIR_PATH = os.path.join("./outputs",output_dir_name)
    # TENSORBOARD_OUTPUT_PATH = os.path.join("./tensorboard",output_dir_name)
    
    os.makedirs(OUTPUT_DIR_PATH,exist_ok=True)
    # os.makedirs(TENSORBOARD_OUTPUT_PATH,exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_checkpoint=True
    resume_from_checkpoint = None
    # resume_from_checkpoint=os.path.join("./outputs/checkpoint.pt")
    pretrained_weight_path = None
    # pretrained_weight_path = "./outputs/log_2025-01-13T22:44:03.724476/best_model.pt"
    test_only=True
    
    logger = create_logger(output_dir=OUTPUT_DIR_PATH,name=__name__,filename="log.txt")
    writer = SummaryWriter(log_dir=OUTPUT_DIR_PATH)
    
    EPOCHS = 100

    
    # URFALL
    # batch_size = 32
    # in_channels=3
    # num_classes = 4
    # n_joints=14
    # seq_len=30
    # dataloaders = build_fukinect_dataloader(split_ratio={'train':0.6,'valid':0.2,'test':0.2}, batch_size=batch_size, seq_len=seq_len, random_seed=SEED)

    # # ImVia
    # EPOCHS = 20
    # batch_size = 256
    # in_channels=3
    # num_classes = 2
    # n_joints=14
    # seq_len=30
    # dataloaders = build_imvia_dataloader(split_ratio={'train':0.6,'valid':0.2,'test':0.2}, batch_size=batch_size, seq_len=seq_len, random_seed=SEED)
    # dataloaders = build_urfall_dataloader(split_ratio={'train':0.6,'valid':0.2,'test':0.2}, batch_size=batch_size, seq_len=seq_len, random_seed=SEED)

    # HARUP
    batch_size = 256
    in_channels=3
    num_classes = 11
    n_joints=14
    seq_len=30
    dataloaders = build_hurup_dataloader(split_ratio={'train':0.6,'valid':0.2,'test':0.2}, batch_size=batch_size, seq_len=seq_len, random_seed=SEED)

    
    
    
   
    model = Model(
        num_class=num_classes,
        num_point=n_joints,
        max_frame=300,
        graph=adjGraph(layout='coco_cut',
                      strategy='uniform'),
        # act_type = 'relu',
        bias = True,
        edge = True,
        block_size=41,
        embed_dim=64,
        n_stage=1,
        act_type='tanh'
    ).to(device)
                     
    
    logger.info(model)
    model.to(device)
    
    logger.info(
        summary(model,input_size=(batch_size, in_channels, seq_len, n_joints),depth=5)
    )
    
    
    optim_info ={
        # "optim_type":"sgd",
        # "optim_params":{
        #     "lr":1e-3,
        #     "momentum":0.9,
        #     },
        
        # "optim_type":"adamw",
        # "optim_params":{
        #     "lr":1e-3,
        #     "betas":(0.9, 0.999),
        #     "eps":1e-08,
        #     },
        
        # "scheduler_type":"cosine",
        # "scheduler_params":{
        #     "t_initial":EPOCHS,
        #     "lr_min":1e-5,
        #     "t_in_epochs":True,
        #     "warmup_t":5,
        #     "warmup_lr_init":1e-4,
        #     },

        "optim_type":"rms",
        "optim_params":{
            "lr":0.001,
            },
        "scheduler_type":None,
        "scheduler_params":None,
        }
    
    optimizer,lr_scheduler = build_optimizer(model,**optim_info)
    loss_fn = torch.nn.CrossEntropyLoss()
    # scaler = torch.cuda.amp.GradScaler()
    scaler = None
    
    start_epoch = 1
    best_acc = 0
    
    if resume_from_checkpoint:
        logger.info("Resume from checkpoint")
        checkpoint = torch.load(resume_from_checkpoint)
        
        start_epoch = checkpoint["epoch"]+1
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["model_weight"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "scaler" in checkpoint : scaler.load_state_dict(checkpoint["scaler"])
        
    if pretrained_weight_path:
        logger.info("Load pretrained weight start")
        model.load_state_dict(torch.load(pretrained_weight_path)["model_weight"])
        logger.info("Load pretrained weight finish")
        time.sleep(1)
        if test_only:
            valid(model,dataloaders["valid"],loss_fn,device,logger,logging_timing=10,epoch=-1)
            test(model,dataloaders["test"],loss_fn,device,logger)
            return
    
    for e in range(start_epoch,EPOCHS+1):
        logger.info(f"[EPOCH] {e}/{EPOCHS}")
        train(model,dataloaders["train"],optimizer,loss_fn,device,logger,writer,logging_timing=10,epoch=e,scaler=scaler)
        val_acc = valid(model,dataloaders["valid"],loss_fn,device,logger,writer,logging_timing=10,epoch=e)
        
        if lr_scheduler != None : lr_scheduler.step(e)
            
        if best_acc < val_acc[0]:
            best_acc = val_acc[0]
            logger.info("[BEST MODEL UPDATED]")
            
            PATH = os.path.join(OUTPUT_DIR_PATH,"best_model.pt")
            torch.save({
                        'model_weight': model.state_dict(),
                        }, PATH)
            
        if save_checkpoint:
            checkpoint = {
                "epoch":e,
                "model_weight":model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler != None else None,
                "scaler":scaler.state_dict() if scaler != None else None,
                "best_acc":best_acc,
            }
            torch.save(checkpoint,os.path.join(OUTPUT_DIR_PATH,"checkpoint.pt"))
            
    
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR_PATH,"best_model.pt"))["model_weight"])
    val_acc = valid(model,dataloaders["valid"],loss_fn,device,logger,logging_timing=10,epoch=-1)
    test(model,dataloaders["test"],loss_fn,device,logger)
        
    
    
    
if __name__=="__main__":
    config=None
    
    run(config)
    # run_for_test()