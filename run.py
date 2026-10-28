import argparse
import os
import torch
import random
import numpy as np
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_pre_train import Exp_Pre_Train
import torch.distributed as dist
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimeHAT')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        choices=['pre_training', 'long_term_forecast'], help='supported task')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--fine_tuning', type=int, required=False, default=0, help='fine_tuning')
    parser.add_argument('--model', type=str, required=False, default='TimeHAT',
                        help='model name, options: [TimeHAT]')
    
    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./', help='root path of the data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--patch_len', type=int, default=96, help='patch size')
    # per_training task
    parser.add_argument('--input_token_len', type=int, default=576, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='output token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_pred_len', type=int, default=96, help='test pred len')
    parser.add_argument('--nonautoregressive', action='store_true', help='nonautoregressive', default=False)
    parser.add_argument('--test_flag', type=str, default='T', help='test domain')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='few shot ratio')
    parser.add_argument('--keep_ratio', type=float, default=0.25, help='keep_ratio')
    # ==============================================================================
    parser.add_argument('--adaptation', action='store_true', help='whether to load the pre-trained model for adaptation/fine-tuning')
    parser.add_argument('--pretrain_model_path', type=str, default='', help='path to the pre-trained model checkpoint for adaptation')
    # forecasting task
    parser.add_argument('--label_len', type=int, default=48, help='start token length') 
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--forecast_mode', choices=['parallel', 'recursive'], default='parallel',
                        help='parallel: patch-local decoding; recursive: one future probe predicts pred_len points, rollout at test')

    parser.add_argument('--recursive_readout', choices=['future_token', 'last_token'], default='future_token',
                        help='recursive head reads one appended future token or the last historical token')
    parser.add_argument('--recursive_window', choices=['fixed', 'growing'], default='fixed',
                        help='recursive inference history: fixed crops to input length; growing keeps all history')

    parser.add_argument('--inference_pred_len', type=int, default=0,
                        help='test horizon; 0 uses pred_len; does not change training/validation')
    parser.add_argument('--forecast_checkpoint', type=str, default='',
                        help='trained downstream checkpoint for test-only inference')

    # model define
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=16, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--full_atten_layers', type=int, default=3, help='full_atten_layers')
    parser.add_argument('--attn_type', type=int, default=0, help='attn_type')
    parser.add_argument('--norm_type', type=int, default=0, help='norm_type')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--learning_rate', type=float, default=8e-4, help='optimizer peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='optimizer weight decay')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='absolute warmup steps for learning rate scheduler')
    parser.add_argument('--max_warmup_ratio', type=float, default=0.05, help='maximum ratio of total steps allowed for warmup to prevent schedule distortion')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--ddp', action='store_true', help='Distributed Data Parallel', default=False)
    parser.add_argument('--dp', action='store_true', help='Data Parallel', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')  
    parser.add_argument('--no_bf16', action='store_false', dest='use_bf16', default=True,  help='disable bfloat16 mixed precision')

    args = parser.parse_args()
    if args.pred_len <= 0 or args.patch_len <= 0 or args.inference_pred_len < 0:
        parser.error('pred_len/patch_len must be positive; inference_pred_len must be nonnegative')
    if args.is_training and args.forecast_checkpoint:
        parser.error('--forecast_checkpoint is for --is_training 0')
    args.data_path=f"{args.data}.csv"
    args.use_gpu = True 

    if args.dp:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]
    elif args.ddp:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
    print('Args in experiment:')
    print(args)
    if args.task_name == 'pre_training':
        setting = '{}_{}_sl{}_pl{}_keepratio{}'.format(
            args.model,
            args.data,
            args.seq_len,
            args.patch_len,
            args.keep_ratio
            )
        exp = Exp_Pre_Train(args) 
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)  
        torch.cuda.empty_cache()

    else:
        # fix_seed = 2025
        # random.seed(fix_seed)
        # torch.manual_seed(fix_seed)
        # np.random.seed(fix_seed)
        Exp = Exp_Long_Term_Forecast

        if args.is_training==1:
            maes, mses = [], []
            mode_tag = args.forecast_mode
            if args.forecast_mode == 'recursive' and args.recursive_readout == 'last_token':
                mode_tag += '_last_token'
            if args.forecast_mode == 'recursive' and args.recursive_window == 'growing':
                mode_tag += '_growing'
            folder_path = f'./0_result_{args.task_name}/fine_tuning/{mode_tag}/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_sl{}_pl{}_ft{}_keepratio{}_patch_len{}'.format(
                    args.model,
                    args.data,
                    args.seq_len,
                    args.pred_len,
                    args.fine_tuning,
                    args.keep_ratio,
                    args.patch_len
                    )
                
                setting += f'_{mode_tag}_itr{ii}'
                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                is_main_rank = not getattr(args, 'ddp', False) or args.local_rank == 0
                if is_main_rank:
                    mae, mse = exp.test(setting)
                    maes.append(mae)  
                    mses.append(mse)  
                    f = open(f"{folder_path}/{args.data}_train{args.pred_len}_test{args.inference_pred_len or args.pred_len}.txt", 'a')
                    f.write('ii:{} mse:{}, mae:{}\n'.format(ii, mse, mae))
                    f.close()  
                    torch.cuda.empty_cache()
                if getattr(args, 'ddp', False):
                    dist.barrier()
            f = open(f"{folder_path}/{args.data}_train{args.pred_len}_test{args.inference_pred_len or args.pred_len}.txt", 'a')
            f.write(setting + "  \n")
            avg_mse=np.mean(mses)
            avg_mae=np.mean(maes)
            f.write('avg:\tmse:{}, mae:{}'.format(avg_mse, avg_mae))
            f.write('\n')
            f.write('\n')
            f.close()
        else:
            if not args.forecast_checkpoint:
                parser.error('--is_training 0 requires --forecast_checkpoint')
            exp = Exp(args)
            mae, mse = exp.test('test_only')
            print(f'Mode={args.forecast_mode}, window={args.recursive_window}, '
                  f'Readout={args.recursive_readout}, '
                  f'Test horizon={args.inference_pred_len or args.pred_len}: mse={mse}, mae={mae}')
