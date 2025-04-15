import os
import numpy as np
import pandas as pd
import random
import datetime
from collections import defaultdict
from termcolor import colored
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time
import yaml
import argparse
import shutil
from tqdm import tqdm
from torchinfo import summary
from sklearn.metrics import (
classification_report,
accuracy_score,
precision_recall_fscore_support,
)

from config import get_cfg_defaults
from dataloader import build_dataloader
from cv_dataloader import build_cv_dataloader

from logger import create_logger
from optimizer import build_optimizer
from models.build_model import build_model



def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_argparse():
    parser = argparse.ArgumentParser(description='Training and Evaluation config')
    parser.add_argument('-cfg','--config', type=str, help='Please set config file (.yaml)',required=True)
    args = parser.parse_args()

    config = get_cfg_defaults()
    config.merge_from_file(args.config)
    config.freeze()
    
    return args, config

def makedir(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath)

def cal_top_k_accuracy(output, target, top_k=[1]):
    output, target = output.cpu(), target.cpu()
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
    for order,(name, param) in enumerate(model.named_parameters()):
        if param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            if writer:
                writer.add_scalar(f"Gradient_Norm/{order}.{name}", grad_norm, epoch)
    
def train(model,dataloader,optimizer,loss_fn,device="cpu",logger=None,writer=None,epoch=None,scaler=torch.cuda.amp.GradScaler(),max_norm=10,top_k=[1,5],accum_iter=16, config=None):    
    model.train()
    running_loss = []
    running_accu = []
    _time_hist =[]
    
    for i,(data,sensor,label) in enumerate(dataloader):
        _start_time = time.time()
        
        data = data.to(device)
        sensor = sensor.to(device)
        label = label.to(device)
        
        if label.dim() == 2:
            label_onehot = label
            
        else:
            label_onehot = torch.nn.functional.one_hot(label, num_classes=config.DATA.NUM_CLASSES)
            label_onehot = label_onehot.float()
        
        with torch.amp.autocast(device.type, dtype=torch.float32):
            pred = model(data, sensor)
            loss = loss_fn(pred,label_onehot)
        
        loss.backward()
        # scaler.scale(loss).backward()
        
        if i%accum_iter==0 or i+1 == len(dataloader):
            # optimizer.step()
            # Unscales the gradients of optimizer's assigned params in-place
            # scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
            # scaler.step(optimizer)
            optimizer.step()
            
            if writer: visualize_weights_and_gradients(model,writer,epoch)
                
            # scaler.update()
            model.zero_grad()
        
        running_loss += [loss.item()]
        running_accu += [cal_top_k_accuracy(pred,label,top_k=top_k)]
        
        _end_time = time.time()
        _time_hist.append(_end_time - _start_time)
        _remaining_sec = (np.mean(_time_hist))*(len(dataloader)-(i+1))

        
        if i % config.LOGGING_TIMING == 0:
            logger.info(f"[{colored('TRAIN','light_green')} Epoch {epoch} : {i}/{len(dataloader)} ({cal_remaining_time(_remaining_sec)})]\t lr : {optimizer.param_groups[0]['lr']:.5f}\tLoss : {running_loss[-1]:.5f}\tAccuracy (Top-1) : {running_accu[-1][0]:.4f}")
                
    logger.info(f"{colored('TRAIN','light_green')} EPOCH SUMMARY: LOSS : {np.mean(running_loss)}\t ACC : {np.mean(running_accu,axis=0)}")
    if writer:
        writer.add_scalar('Loss/train', np.mean(running_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(running_accu,axis=0)[0], epoch)
    
@torch.no_grad()
def valid(model,dataloader,loss_fn,device,logger=None,writer=None,epoch=None,top_k=[1,5],config=None):
    model.eval()
    
    prediction = []
    labels = []
    running_loss = []
    running_accu = []
    _time_hist =[]
    for i,(data,sensor,label) in enumerate(dataloader):
        _start_time = time.time()
        data = data.to(device)
        sensor = sensor.to(device)
        label = label.to(device)
        
        if label.dim() == 2:
            label_onehot = label
        else:
            label_onehot = torch.nn.functional.one_hot(label, num_classes=config.DATA.NUM_CLASSES)
            label_onehot = label_onehot.float()


        pred = model(data, sensor)
        # pred = model(sensor)
        
        loss = loss_fn(pred,label)

        model.zero_grad()
        
        running_loss += [loss.item()]
        running_accu += [cal_top_k_accuracy(pred,label,top_k=top_k)]

        _end_time = time.time()
        _time_hist.append(_end_time - _start_time)
        _remaining_sec = (np.mean(_time_hist))*(len(dataloader)-(i+1))
        
        if i % config.LOGGING_TIMING == 0:
            logger.info(f"[{colored('VALID','light_yellow')} Epoch {epoch} : {i}/{len(dataloader)} ({cal_remaining_time(_remaining_sec)})]\tAccuracy (Top-1) : {running_accu[-1][0]:.4}")

        prediction += [pred]
        labels += [label]
        
    prediction = torch.cat(prediction)
    labels = torch.cat(labels)
            
    logger.info(f"{colored('VALID','light_yellow')} EPOCH SUMMARY: LOSS : {np.mean(running_loss)}\tACC : {cal_top_k_accuracy(prediction,labels,top_k=top_k)}")
    if writer:
        writer.add_scalar('Loss/val', np.mean(running_loss), epoch)
        writer.add_scalar('Accuracy/val', cal_top_k_accuracy(prediction,labels,top_k=top_k)[0], epoch)
    
    return cal_top_k_accuracy(prediction,labels,top_k=top_k)

@torch.no_grad()
def test(model,dataloader,loss_fn,device,logger=None,epoch=None,top_k=[1,5],config=None):
    model.eval()

    prediction = []
    labels = []
    running_loss = []
    running_accu = []
    for i,(data,sensor,label) in enumerate(dataloader):
        data = data.to(device)
        sensor = sensor.to(device)
        label = label.to(device)
        
        if label.dim() == 2:
            label_onehot = label
        else:
            label_onehot = torch.nn.functional.one_hot(label, num_classes=config.DATA.NUM_CLASSES)
            label_onehot = label_onehot.float()


        pred = model(data, sensor)
        # pred = model(sensor)
            
            
        loss = loss_fn(pred,label)

        model.zero_grad()
        
        running_loss += [loss.item()]
        running_accu += [cal_top_k_accuracy(pred,label,top_k=top_k)]
        
        if i % config.LOGGING_TIMING == 0:
            logger.info(f"[{colored('TEST','red')} : {i}/{len(dataloader)}]\tAccuracy (Top-1) : {running_accu[-1][0]:.4}")

        prediction += [pred]
        labels += [label]
        
    _prediction = torch.cat(prediction)
    _labels = torch.cat(labels)
    
    logger.info(f"{colored('TEST','red')} EPOCH SUMMARY: LOSS : {np.mean(running_loss)}\t ACC : {cal_top_k_accuracy(_prediction,_labels,top_k=top_k)}")
    
    prediction = torch.argmax(_prediction, dim=1).to('cpu').detach().numpy().copy()
    labels = _labels.to('cpu').detach().numpy().copy() if _labels.dim()==1 else torch.argmax(_labels,dim=1).to('cpu').detach().numpy().copy()
    logger.info(f"[[[[[Classification Report]]]]]\n{classification_report(labels,prediction,digits=5)}")

    return cal_top_k_accuracy(_prediction,_labels,top_k=top_k), precision_recall_fscore_support(labels,prediction,average='macro')
    
    
    
    
def run(config=None):
    global OUTPUT_DIR_PATH

    # get random_seed for reproducibility
    SEED = config.SEED
    fix_seed(config.SEED)

    # make output dir
    output_dir_name = config.LOG_DIR if config.LOG_DIR != None else "log_"+datetime.datetime.now().isoformat()
    OUTPUT_DIR_PATH = os.path.join("./outputs",output_dir_name)
    makedir(OUTPUT_DIR_PATH)

    
    save_checkpoint=config.SAVE_CHECKPOINT
    resume_from_checkpoint=config.RESUME_FROM
    pretrained_weight_path=config.PRETRAINED_WEIGHT_PATH
    test_only=config.TEST_ONLY
    device = torch.device(config.DEVICE)
    EPOCHS = config.TRAIN.EPOCHS

    
    
    cv_dataloaders = build_cv_dataloader(config)
    logger = create_logger(output_dir=OUTPUT_DIR_PATH, name=__name__, filename="log.txt")
    writer = SummaryWriter(log_dir=OUTPUT_DIR_PATH) if config.TENSORBOARD_LOG else None

    cv_precision_recall_f1 = defaultdict(list)
    
    for i,dataloaders in enumerate(cv_dataloaders):
        logger.info(f"***************** START {i} fold CV *****************")
        model = build_model(config)
        optimizer,lr_scheduler = build_optimizer(model,config)
        
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.TRAIN.LABEL_SMOOTHING)
        scaler = torch.cuda.amp.GradScaler() if config.TRAIN.USE_SCALER else None
        
    
        logger.info(f"Using device {device}.")    
        logger.info(model)
        model.to(device)
        
        
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
            scaler.load_state_dict(checkpoint["scaler"])
            
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
            train(model,dataloaders["train"],optimizer,loss_fn,device,logger,writer,epoch=e, scaler=scaler,max_norm=config.TRAIN.MAX_NORM, top_k=config.TOP_K, accum_iter=config.TRAIN.ACCUM_ITER, config=config)
            val_acc = valid(model,dataloaders["valid"],loss_fn,device,logger,writer,epoch=e, top_k=config.TOP_K, config=config)
            
            if lr_scheduler:
                lr_scheduler.step(e)
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
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                    "scaler":scaler.state_dict(),
                    "best_acc":best_acc,
                }
                torch.save(checkpoint,os.path.join(OUTPUT_DIR_PATH,"checkpoint.pt"))
                
        
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR_PATH,"best_model.pt"))["model_weight"])
        val_acc = valid(model,dataloaders["valid"],loss_fn,device,logger,epoch=-1, top_k=config.TOP_K, config=config)
        test_acc, precision_recall_f1 = test(model,dataloaders["test"],loss_fn,device,logger, top_k=config.TOP_K, config=config)
        logger.info(f"***************** END {i} fold CV *****************")
        
        cv_precision_recall_f1["precision"].append(precision_recall_f1[0])
        cv_precision_recall_f1["recall"].append(precision_recall_f1[1])
        cv_precision_recall_f1["f1"].append(precision_recall_f1[2])
        cv_precision_recall_f1["accuracy"].append(test_acc[0])

    pd.DataFrame(cv_precision_recall_f1).to_csv(os.path.join(OUTPUT_DIR_PATH,"precision_recall_f1.csv"))
    return val_acc, test_acc
    
    
if __name__=="__main__":
    args,config = make_argparse()    
    
    _, _ = run(config)
    
    with open(os.path.join(OUTPUT_DIR_PATH,"config.yaml"), "w") as f:
        f.write(config.dump())