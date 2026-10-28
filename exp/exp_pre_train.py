import os
import time
import warnings
import glob
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from torch.optim.lr_scheduler import LambdaLR

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjust_learning_rate, visual
from utils.metrics import metric
import torch.distributed as dist

warnings.filterwarnings('ignore')


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Foundation pre-training scheduler: Fixed-step linear warmup + cosine decay to min_lr_ratio
    """
    def lr_lambda(current_step):
        
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

class Exp_Pre_Train(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pre_Train, self).__init__(args)

    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self.args.gpu
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)
            
        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_size = batch_x.size(0)
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.args.use_bf16):
                    loss = self.model(batch_x, batch_x_mark, batch_x, batch_x_mark).mean()
                    
                total_loss += loss * batch_size
                total_samples += batch_size

        
        if self.args.ddp:
            
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            
        avg_loss = (total_loss / total_samples).item()
        self.model.train()
        return avg_loss

    def train(self, setting):
        print_model_params(self.model)  
        
        print("ATTENTION!!! Begin Dataloader")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        total_steps = self.args.train_epochs * train_steps 

        peak_lr = self.args.learning_rate
        weight_decay = self.args.weight_decay
        
        
        warmup_steps = min(self.args.warmup_steps, int(self.args.max_warmup_ratio * total_steps))
        
        model_optim = torch.optim.AdamW(self.model.parameters(), lr=peak_lr, weight_decay=weight_decay)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=model_optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=0.1  
        )
        
        print(f"--- Pre-training Configuration Started ---")
        print(f"Total Steps: {total_steps}")
        print(f"Warmup Steps: {warmup_steps}")
        print(f"Peak LR: {peak_lr}")
        print(f"Weight Decay: {weight_decay}")
        print(f"Steps per Epoch: {train_steps}")
        print(f"Mixed Precision (BF16): Enabled")
        print(f"----------------------")

        start_epoch = 0
        global_step = 0
        steps_in_current_epoch = 0 
        
        checkpoints = glob.glob(os.path.join(path, 'checkpoint_step_*.pth'))
        
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(x.split('_step_')[-1].split('.pth')[0]))
            latest_ckpt_path = checkpoints[-1]
            
            print(f"\n[Resume] Found historical Checkpoint: {latest_ckpt_path}")
            print(f"[Resume] Restoring model, optimizer, and scheduler states...")
            
            checkpoint = torch.load(latest_ckpt_path, map_location=self.device)
            
            
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
                
            
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['step']
            
            steps_in_current_epoch = global_step % train_steps
            
            print(f"[Resume] Success! Resuming training from Epoch {start_epoch+1}, Global Step {global_step}.")
            print(f"[Dataloader] Fast-forwarding to skip the first {steps_in_current_epoch} batches in this epoch...\n")
        else:
            print(f"\nNo historical Checkpoint detected, starting fresh pre-training...\n")

                
        time_now = time.time()
        is_main_rank = not self.args.ddp or self.args.local_rank == 0
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            if self.args.ddp and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if epoch == start_epoch and i < steps_in_current_epoch:
                    continue
                
                global_step += 1
                iter_count += 1
                
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.args.use_bf16):
                    loss = self.model(batch_x, batch_x_mark, batch_x, batch_x_mark) 
                    loss = loss.mean()
                
                train_loss.append(loss.item())
                
                if self.args.dp:
                    torch.cuda.synchronize()
                
                if (i + 1) % 100 == 0:
                    current_lr = model_optim.param_groups[0]['lr']
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | lr: {3:.7f}".format(
                        i + 1, epoch + 1, loss.item(), current_lr))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (total_steps - global_step)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
                scheduler.step()

                is_last_step = (epoch == self.args.train_epochs - 1) and (i == len(train_loader) - 1)
                if (global_step % 5000 == 0 or is_last_step) and is_main_rank:
                    self.save_checkpoint_step(global_step, epoch, model_optim, scheduler, path, keep_last=30)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
        

    def save_checkpoint_step(self, step, epoch, model_optim, scheduler, path, keep_last=30):
        
        if self.args.ddp and self.args.local_rank != 0:
            return

        
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        param_grad_dic = {k: v.requires_grad for (k, v) in model_to_save.named_parameters()}
        state_dict = model_to_save.state_dict()
        
        
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]
                
        state = {
            'step': step, 'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': model_optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        
        checkpoint_name = f'checkpoint_step_{step}.pth'
        save_path = os.path.join(path, checkpoint_name)
        torch.save(state, save_path)
        print(f"[Rank 0] Checkpoint successfully saved at step {step}: {save_path}")
        
        
        checkpoints = glob.glob(os.path.join(path, 'checkpoint_step_*.pth'))
        checkpoints.sort(key=lambda x: int(x.split('_step_')[-1].split('.pth')[0]))
        
        if len(checkpoints) > keep_last:
            for old_ckpt in checkpoints[:-keep_last]:
                os.remove(old_ckpt)
                print(f"[Space Cleanup] Deleted old checkpoint: {old_ckpt}")


def print_model_params(model):
    """Print model parameter statistics"""
    print("=== Model Parameter Statistics ===")
    total_params = 0
    trainable_params = 0
    print(f"{'Layer Name':<30} {'Shape':<25} {'Params':<15} {'Trainable'}")
    print("-" * 85)
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
            
        short_name = name if len(name) < 28 else name[:25] + "..."
        print(f"{short_name:<30} {str(list(param.shape)):<25} {param_count:<15,} {param.requires_grad}")
    print("-" * 85)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    return total_params, trainable_params