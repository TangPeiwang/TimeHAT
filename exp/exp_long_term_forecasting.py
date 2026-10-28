import os
import copy
import time
import warnings
import math
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from collections import Counter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual
from utils.metrics import metric
import torch.distributed as dist
warnings.filterwarnings('ignore')


def get_finetune_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Downstream fine-tuning scheduler: Strictly executes linear warmup for a specified number of steps (5%), 
    followed by cosine decay.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        
        if getattr(self.args, 'ddp', False):
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self.args.gpu
            
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        if getattr(self.args, 'ddp', False):
            model = model.to(self.device)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.args.local_rank])
        elif getattr(self.args, 'dp', False):
            model = nn.DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            model = model.to(self.device)
            
        if getattr(self.args, "adaptation", False):
            print(f"Loading pre-trained model for adaptation from: {self.args.pretrain_model_path}")
            self.load_matching_layers(model, self.args.pretrain_model_path)
            
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        backbone_params = []
        head_params = []
        

        for name, param in self.model.named_parameters():
            
            if 'decoder' in name or 'projection' in name or 'head' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        # Backbone LR corresponds to [1e-5, 3e-5] in the paper
        lr_backbone = getattr(self.args, 'lr_backbone', 1e-5) 
        # Head LR corresponds to [5e-5, 1e-4] in the paper
        lr_head = self.args.learning_rate 
        weight_decay = getattr(self.args, 'weight_decay', 1e-4)
        
        optim_groups = [
            {'params': backbone_params, 'lr': lr_backbone, 'name': 'backbone'},
            {'params': head_params, 'lr': lr_head, 'name': 'head'}
        ]
        
        print(f"Optimizer Groups: Backbone LR={lr_backbone}, Head LR={lr_head}")
        
        model_optim = optim.AdamW(optim_groups, weight_decay=weight_decay)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def load_matching_layers(self, model, checkpoint_path, ignore_mismatch=True):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint)
        
        clean_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            clean_key = k[7:] if k.startswith('module.') else k
            clean_pretrained_dict[clean_key] = v
            
        target_model = model.module if hasattr(model, "module") else model
        model_dict = target_model.state_dict()
        
        matched_dict = {k: v for k, v in clean_pretrained_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        
        reset_decoder = (getattr(target_model, 'forecast_mode', 'parallel') == 'recursive'
                         and target_model.patch_len != target_model.pred_len)
        if reset_decoder:
            # Adaptation retains the newly initialized horizon-sized decoder as a unit.
            matched_dict = {k: v for k, v in matched_dict.items() if not k.startswith('decoder.')}
            print(f'Fresh recursive prediction head: D -> {target_model.pred_len}; '
                  f'pretraining patch length is {target_model.patch_len}.')
        else:
            decoder_keys = {k for k in model_dict if k.startswith('decoder.')}
            if decoder_keys and decoder_keys <= matched_dict.keys():
                print('Loaded the complete pretrained recovery head (weight and bias).')
            elif decoder_keys:
                raise ValueError('Pretrained recovery head is missing or incompatible; '
                                 'check checkpoint and model dimensions')

        if len(matched_dict) == 0:
            raise ValueError("Fatal error: 0 pre-trained weights matched! Please check the prefix unwrapping logic!")
            
        print(f"Successfully loaded layers: {len(matched_dict)}/{len(model_dict)}")
        
        model_dict.update(matched_dict)
        target_model.load_state_dict(model_dict)
        
        return model


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        total_samples = torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_size = batch_x.size(0)
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if getattr(self.args, 'use_amp', False):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=getattr(self.args, 'use_bf16', True)):
                        outputs = self.model(batch_x, batch_x_mark, None, None)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                
                total_loss += loss * batch_size
                total_samples += batch_size


        if getattr(self.args, 'ddp', False):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            
        avg_loss = (total_loss / total_samples).item()
        self.model.train()
        return avg_loss

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        time_now = time.time()
        
        train_steps = len(train_loader)
        # Paper requirements: max 20 epochs, patience set to 5
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        total_steps = self.args.train_epochs * train_steps
        warmup_steps = int(0.05 * total_steps)
        
        print(f"--- Fine-tuning configuration started ---")
        print(f"Total Steps: {total_steps}")
        print(f"Warmup Steps (5%): {warmup_steps}")
        
        scheduler = get_finetune_schedule_with_warmup(
            optimizer=model_optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        is_main_rank = not (hasattr(self.args, 'ddp') and self.args.ddp) or self.args.local_rank == 0
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            if hasattr(self.args, 'ddp') and self.args.ddp and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=getattr(self.args, 'use_bf16', True)):
                        
                        outputs = self.model(batch_x, batch_x_mark, None, None)
                        
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_x, batch_x_mark, None, None)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # Display Backbone and Head LRs separately in the logs
                    lr_bk = model_optim.param_groups[0]['lr']
                    lr_hd = model_optim.param_groups[1]['lr']
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | LR(bk): {3:.7f} | LR(hd): {4:.7f}".format(
                        i + 1, epoch + 1, loss.item(), lr_bk, lr_hd))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            early_stop_flag = 0
            if is_main_rank:
                print("Epoch: {0}, Steps: {1} | Vali Loss: {2:.7f}".format(
                    epoch + 1, train_steps, vali_loss))
                
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                early_stopping(vali_loss, model_to_save, path)

                if early_stopping.early_stop:
                    early_stop_flag = 1
                    print("Early stopping triggered on validation set.")
                    
            if getattr(self.args, 'ddp', False):
                
                stop_tensor = torch.tensor(early_stop_flag, device=self.device)
                dist.broadcast(stop_tensor, src=0)
                early_stop_flag = stop_tensor.item()
                
            if early_stop_flag:
                break
                
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting):
        if getattr(self.args, 'ddp', False) and self.args.local_rank != 0:
            return 0.0, 0.0
            
        horizon = getattr(self.args, 'inference_pred_len', 0) or self.args.pred_len
        test_args = copy.copy(self.args)
        test_args.pred_len = horizon
        test_data, test_loader = data_provider(test_args, 'test')
        print('loading model')
        
        
        target_model = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = getattr(self.args, 'forecast_checkpoint', '') or os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        target_model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if getattr(self.args, 'use_amp', False):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=getattr(self.args, 'use_bf16', True)):
                        outputs = target_model.predict(batch_x, horizon=horizon)
                else:
                    outputs = target_model.predict(batch_x, horizon=horizon)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -horizon:, f_dim:]
                batch_y = batch_y[:, -horizon:, f_dim:]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        return mae, mse