import argparse
import os
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
import random
import numpy as np

fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MADLLM')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model', type=str, required=True, default='GPT4TS')
parser.add_argument('--model_name', type=str, required=True, default='PSM', help='the dir of stored model')

# data loader
parser.add_argument('--data', type=str, required=True, default='PSM', help='dataset type')
parser.add_argument('--root_path', type=str, default='./all_datasets/PSM', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# anomaly detection task
parser.add_argument('--seq_len', type=int, default=100, help='input sequence length')
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')



# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# patching
parser.add_argument('--patch_size', type=int, default=25)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--gpt_layers', type=int, default=6)
parser.add_argument('--ln', type=int, default=0)
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--weight', type=float, default=0)
parser.add_argument('--percent', type=int, default=5)

# skip embedding
parser.add_argument('--use_skip_embedding', type=int, default=0)

# feature embedding
parser.add_argument('--use_feature_embedding', type=int, default=0)
parser.add_argument('--nb_random_samples', type=int, default=10)
parser.add_argument('--feature_lr', type=float, default=0.001)
parser.add_argument('--feature_epochs', type=int, default=10)
parser.add_argument('--channels', type=int, default=25)

# prompt embedding
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--prompt_len', type=int, default=5)
parser.add_argument('--pool_size', type=int, default=10)
parser.add_argument('--use_prompt_pool', type=int, default=0)

parser.add_argument('--visualize', type=int, default=0)

parser.add_argument('--few_shot', type=int, default=0)

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Anomaly_Detection

if not args.visualize and args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = 'pa{}_lr{}_{}_{}_sl{}_dm{}_df{}_{}_se{}_fe{}_pp{}_top{}_pl{}_ps{}_nrs{}_flr{}_fepo{}_ch{}_fs{}'.format(
                    args.patch_size,
                    args.learning_rate,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.d_model,
                    args.d_ff,
                    ii,
                    args.use_skip_embedding,
                    args.use_feature_embedding,
                    args.use_prompt_pool,
                    args.top_k,
                    args.prompt_len,
                    args.pool_size,
                    args.nb_random_samples,
                    args.feature_lr,
                    args.feature_epochs,
                    args.channels,
                    args.few_shot)

        if args.data == "SMD":
            total_train_average_t = 0.0
            SMD_file_list = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8',
                             '2-1', '2-2', '2-3', '2-4', '2-5', '2-6', '2-7', '2-8', '2-9',
                             '3-1', '3-2', '3-3', '3-4', '3-5', '3-6', '3-7', '3-8', '3-9', '3-10', '3-11']
            accuracys, precisions, recalls, f_scores, auc_scores = [], [], [], [], []
            
            for file in SMD_file_list:
                file = "SMD" + file
                args.data = file
                exp = Exp(args)
                setting2 = 'pa{}_lr{}_{}_{}_sl{}_dm{}_df{}_{}_se{}_fe{}_pp{}_top{}_pl{}_ps{}_nrs{}_flr{}_fepo{}_ch{}'.format(
                    args.patch_size,
                    args.learning_rate,
                    args.model,
                    file,
                    args.seq_len,
                    args.d_model,
                    args.d_ff,
                    ii,
                    args.use_skip_embedding,
                    args.use_feature_embedding,
                    args.use_prompt_pool,
                    args.top_k,
                    args.prompt_len,
                    args.pool_size,
                    args.nb_random_samples,
                    args.feature_lr,
                    args.feature_epochs,
                    args.channels)
                setting2 = os.path.join(setting, setting2)
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting2))
                _, train_average_t = exp.train(setting2)
                total_train_average_t += train_average_t

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting2))
                accuracy, precision, recall, f_score, auc_score = exp.test(setting2)
                accuracys.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f_scores.append(f_score)
                auc_scores.append(auc_score)
                torch.cuda.empty_cache()
            train_average_t = total_train_average_t / len(SMD_file_list)
            train_average_t = train_average_t / 1000
            print(f"SMD average train time: {train_average_t}s")
            accuracy, precision, recall, f_score, auc_score = np.mean(accuracys), np.mean(precisions), np.mean(recalls), np.mean(f_scores), np.mean(auc_scores)
            setting = 'pa{}_lr{}_{}_"SMD_average"_sl{}_dm{}_df{}_{}_se{}_fe{}_pp{}_top{}_pl{}_ps{}_nrs{}_flr{}_fepo{}_ch{}'.format(
                    args.patch_size,
                    args.learning_rate,
                    args.model,
                    args.seq_len,
                    args.d_model,
                    args.d_ff,
                    ii,
                    args.use_skip_embedding,
                    args.use_feature_embedding,
                    args.use_prompt_pool,
                    args.top_k,
                    args.prompt_len,
                    args.pool_size,
                    args.nb_random_samples,
                    args.feature_lr,
                    args.feature_epochs,
                    args.channels)
            print("Mean Value:")
            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
                accuracy, precision,
                recall, f_score, auc_score))
            f = open("result_anomaly_detection.txt", 'a')
            f.write(setting + "  \n")
            f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}, Train Time: {:0.4f}s".format(
                accuracy, precision,
                recall, f_score, auc_score, train_average_t))
            f.write('\n')
            f.write('\n')
            f.close()
        else:
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
elif not args.visualize:
    exp = Exp(args)  # set experiments
    if args.data == "SMD":
        path = os.path.join('./checkpoints', args.model_name)
        for dir in os.listdir(path):
            dir = os.path.join(args.model_name, dir)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(dir))
            exp.test(dir, test=1)
            torch.cuda.empty_cache()
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.model_name))
        # exp.test(setting, test=1)
        exp.test(args.model_name, test=1)
        torch.cuda.empty_cache()
else:
    args.data = "SMD1-5"
    exp = Exp(args)
    ii=0
    setting = 'pa{}_lr{}_{}_{}_sl{}_dm{}_df{}_{}_se{}_fe{}_pp{}_top{}_pl{}_ps{}_nrs{}_flr{}_fepo{}_ch{}'.format(
                    args.patch_size,
                    args.learning_rate,
                    args.model,
                    args.data,
                    args.seq_len,
                    args.d_model,
                    args.d_ff,
                    ii,
                    args.use_skip_embedding,
                    args.use_feature_embedding,
                    args.use_prompt_pool,
                    args.top_k,
                    args.prompt_len,
                    args.pool_size,
                    args.nb_random_samples,
                    args.feature_lr,
                    args.feature_epochs,
                    args.channels)
    exp.draw(setting, args.enc_in, True)