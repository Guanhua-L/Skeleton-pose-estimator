import argparse
import logging
import logging.config
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import math
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models import (alexnet, googlenet, inception_v3, resnet18,
                                resnet34, resnet50)
from tqdm import tqdm

# from FIA_dataset.FIA_dataset_2DCNN import FIA_dataset
from FIA_dataset.FIA_dataset_CNNLSTM import FIA_dataset
from model import MARS, CNN_LSTM, Transformer, CNN
from util import get_config_2DCNN, get_config_CNNLSTM, get_distance, get_mae, init_device
# import resnet # 3DCNN

# logger = logging.getLogger()

transform_alexnet = T.Resize(64)
transform_googlenet = T.Resize(16)
transform_inception_v3 = T.Resize(304)

torch.set_printoptions(precision=5, sci_mode=False)

def compute_performance(pred, labels):
    mse = torch.mean((pred - labels) ** 2)
    return mse 

class customLoss(torch.nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()

    def forward(self, output, target, uncertainty):
        distance_error = output - target
        diff = torch.pow( (distance_error/uncertainty), 2 )
        # print('diff: ', diff)
        # print('uncertainty: ', uncertainty)
        un = torch.log(torch.pow(uncertainty,2))
        # print('uncertainty log: ', un )
        final = torch.mean( torch.pow(distance_error, 2) ) + 0.001*torch.mean( diff + un )
        # print('final: ', final)
        return final
    
class Trainer():
    def __init__(self, args: argparse.Namespace, result_path: Path):
        self.args, self.result_path = args, result_path
        # self.lr, self.batch_size, self.epoch, self.weight_decay, self.lr_decay = get_config_2DCNN(result_path, args.model)
        self.lr, self.batch_size, self.epoch, self.weight_decay, self.lr_decay, self.input_size, self.hidden_size, self.num_layers, self.dropout, self.bidirectional, self.window = get_config_CNNLSTM(result_path, args.model)
        self.device = init_device(args.gpu)
        torch.cuda.set_device(self.device)
        self.train_loader, self.test_loader = self.init_data()

    def init_data(self) -> 'tuple[DataLoader, DataLoader]':
        if self.args.test:
            train_set = FIA_dataset(
                subjects=list(filter(lambda subject: subject != self.args.test, range(1, 25))),
                cr=self.args.cr,
                window=self.window  # CNNLSTM
            )
            test_set = FIA_dataset(
                subjects=[self.args.test],
                cr=self.args.cr,
                window=self.window  # CNNLSTM
            )
            train_loader = DataLoader(
                dataset=train_set,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=10,
                drop_last=True,
            )
            test_loader = DataLoader(
                dataset=test_set,
                shuffle=False,
                batch_size=self.batch_size,
                num_workers=10,
                drop_last=True,
            )
        else:
            '''Random 8:2 Split'''
            dataset = FIA_dataset(subjects=list(range(1, 25)), cr=self.args.cr, window=self.window)
            indices = list(range(len(dataset)))
            split = int(np.floor(.8 * len(dataset)))
            np.random.shuffle(indices)
            train_indices, test_indices = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_indices)
            test_sampler = SubsetRandomSampler(test_indices)
            train_loader = DataLoader(
                dataset=dataset,
                sampler=train_sampler,
                batch_size=self.batch_size,
                num_workers=10,
                drop_last=True,
            )
            test_loader = DataLoader(
                dataset=dataset,
                sampler=test_sampler,
                batch_size=self.batch_size,
                num_workers=10,
                drop_last=True,
            )
            '''5-fold 8:2 Split'''
            # train_dataset = FIA_dataset(subjects=list(range(1, 25)), cr=self.args.cr, fold=self.args.fold, train_or_test='train', window=self.window)
            # test_dataset = FIA_dataset(subjects=list(range(1, 25)), cr=self.args.cr, fold=self.args.fold, train_or_test='test', window=self.window)
            # train_loader = DataLoader(
            #     dataset=train_dataset,
            #     batch_size=self.batch_size,
            #     shuffle=True,
            #     num_workers=10,
            #     drop_last=True
            # )
            # test_loader = DataLoader(
            #     dataset=test_dataset,
            #     batch_size=self.batch_size,
            #     shuffle=True,
            #     num_workers=10,
            #     drop_last=True
            # )
        return train_loader, test_loader

    def resize_data(self, data):
        # if self.args.model == 'alexnet':
        #     return transform_alexnet(data)
        # elif self.args.model == 'googlenet':
        #     return transform_googlenet(data)
        # elif self.args.model == 'inception_v3':
        #     return transform_inception_v3(data)
        # return data
        if self.args.model == 'MARS':
            return data # 8*8
        return transform_alexnet(data) # 64*64

    def train(self):
        # model = globals()[self.args.model](num_classes=39).to(self.device)  # 2DCNN
        # model = resnet.resnet34(num_classes=39, shortcut_type='B', sample_size=8, sample_duration=8).to(self.device) # 3DCNN
        # model = CNN_LSTM(globals()[self.args.model], self.input_size, self.hidden_size, self.num_layers, 39, self.dropout, self.bidirectional, self.device, self.window).to(self.device)
        #! test other model
        # model = CNN_LSTM(globals()[self.args.model], self.input_size, self.hidden_size, self.num_layers, 39, self.dropout, self.bidirectional, self.device, self.window).to(self.device)
        model_cnn = CNN(globals()[self.args.model], self.input_size, self.hidden_size, self.num_layers, 39, self.dropout, self.bidirectional, self.device, self.window).to(self.device)
        model = Transformer(src_vocab_size=self.batch_size*39, tgt_vocab_size=39, d_model=96, num_heads=6, num_layers=3, d_ff=512, max_seq_length=2048, dropout=0.25, device=self.device).to(self.device)
        # model = CNN_GRU(globals()[self.args.model], self.input_size, self.hidden_size, self.num_layers, 39, self.dropout, self.bidirectional, self.device, self.window).to(self.device)
        
        # model load pt
        # checkpoint = Path(f'/mnt/data/guanhua/SPE/result/20240306_170605_CR_test7_resnet34/checkpoints/8.pt') #20240308_125621_CR_test7_resnet34
        # checkpoint = Path(f'/mnt/data/guanhua/SPE/result/20240306_171631_CR_8:2_fold1_resnet34/checkpoints/12.pt') #20240308_125903_CR_8:2_fold1_resnet34
        # model.load_state_dict(torch.load(checkpoint))

        uncertainty = False
        loss_function = torch.nn.MSELoss()
        loss_function_un = customLoss()
        # if uncertainty:
        #     loss_function = customLoss()
        # else:
        #     loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model_cnn.parameters(), self.lr, weight_decay=self.weight_decay, betas=[0.5, 0.999], amsgrad=False)
        lr_decayer = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)
        #!transformer
        optimizer2 = torch.optim.Adam(model.parameters(), self.lr, weight_decay=self.weight_decay, betas=[0.5, 0.999], amsgrad=False)
        lr_decayer2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, self.lr_decay)

        min_test_distance_avg = 999999

        variance = 0.0
        for epoch in range(1, self.epoch + 1):
            
            '''Train'''
            model.train()
            train_loss, train_mae, train_distance = 0., torch.zeros((self.batch_size * len(self.train_loader), 13)), torch.zeros((self.batch_size * len(self.train_loader), 13))
            train_start_time = time.time()
            pbar = tqdm(self.train_loader)
            # if epoch < 2:
            #     print(f'Skip epoch:{epoch}')
            #     continue
            for i, (data, labels) in enumerate(pbar):
                # data = self.resize_data(data)
                data = data.to(self.device)
                labels = labels.to(self.device)
                # print(f'data size: {data.shape}, labels size: {labels.shape}')
                # exit()
                #? uncertainty mc dropout
                if uncertainty:
                    num_samples = 10
                    preds = []
                    for _ in range(num_samples):
                        pred = model_cnn(data)
                        pred = pred.view(256, 39)
                        # print(pred)
                        preds.append(pred)
                    preds = torch.stack(preds)

                    pred = torch.mean(preds, dim=0)
                    variance = preds.var(0)
                else:
                    pred = model_cnn(data)
                    pred = pred.view(256, 39)
                # print('pred: ', pred)
                # print(pred.shape)
                # exit()
                loss = loss_function(pred, labels)

                if uncertainty:
                    pred = model(pred, variance)
                else:
                    pred = model(pred)
                pred = pred[:, -1, :].to(self.device)
                # exit()
                # num_samples = 10
                # preds = []
                # for _ in range(num_samples):
                #     pred = model(data)
                #     preds.append(pred)
                # preds = torch.stack(preds)

                # pred = torch.mean(preds, dim=0)
                # variance = preds.var(0).cuda()
                # print(f'pred size: {pred.shape}, labels size: {labels.shape}')
                # pred2 = transformer(pred, variance)
                # pred2 = transformer(pred)
                # pred2 = pred2.to(self.device)
                # # print(f'pred2 size: {pred2.shape}, labels size: {labels.shape}')
                # loss2 = loss_function(pred2, labels)
                # optimizer2.zero_grad()
                # loss2.backward(retain_graph=True)
                # optimizer2.step()
                # pred = model(data)
                
                # torch.set_printoptions(precision=10, sci_mode=False)
                # print('preds: ', preds)
                # print('mean: ', pred)
                # print('var: ', variance)
                # exit()
                # print('pred:', pred.shape)
                # print(f'var: {variance.shape}')
                # print('meanpred: ', mean_prediction.mean(0))
                # print('var:',variance)
                
                # print('diff:', np.array(mean_prediction.tolist())-np.array(labels.tolist()))
                # pred = mean_prediction
                # print('pred:',pred.tolist())
                # print('label:',labels.tolist())
                # print('mean:',mean_prediction)
                # print('var:',[t.tolist() for t in variance])
                # print('var:', variance.tolist())
                # exit()
                

                if self.args.model in ('googlenet', 'inception_v3'):
                    pred = pred.logits

                # print(f'pred: {pred.shape}, labels: {labels.shape}')
                if uncertainty:
                    loss2 = loss_function_un(pred, labels, variance)
                else:
                    loss2 = loss_function(pred, labels)
                # if loss < 0:
                #     train_loss_error_counter += 1
                # print('pred2: ', pred)
                # print('loss: ', loss)
                # print('var: ', variance)
                # exit()
                train_loss += loss2.item()

                train_mae[i * self.batch_size:(i + 1) * self.batch_size, :] = get_mae(pred, labels)
                # print('train mae: ', train_mae.mean(0).shape)
                # exit()
                train_distance[i * self.batch_size:(i + 1) * self.batch_size, :] = get_distance(pred, labels)

                optimizer.zero_grad()
                optimizer2.zero_grad()
                loss.backward(retain_graph=True)
                loss2.backward()

                optimizer.step()
                optimizer2.step()

                # exit()

            train_end_time = time.time()
            train_time = train_end_time - train_start_time

            

            '''Test'''
            model.eval()
            with torch.no_grad():
                test_loss, test_mae, test_distance = 0., torch.zeros((self.batch_size * len(self.test_loader), 13)), torch.zeros((self.batch_size * len(self.test_loader), 13))
                test_mae2, test_distance2 = torch.zeros((self.batch_size * len(self.test_loader), 13)), torch.zeros((self.batch_size * len(self.test_loader), 13))
                inference_time = 0.
                pbar = tqdm(self.test_loader)
                for i, (data, labels) in enumerate(pbar):
                    # data = self.resize_data(data)
                    data = data.to(self.device)
                    labels = labels.to(self.device)

                    start_time = time.time()

                    if uncertainty:
                        num_samples = 10
                        preds = []
                        for _ in range(num_samples):
                            pred = model_cnn(data)
                            pred = pred.view(256, 39)
                            preds.append(pred)
                        preds = torch.stack(preds)
                        pred = torch.mean(preds, dim=0)
                        variance = preds.var(0)

                    else:
                        pred = model_cnn(data)
                        pred = pred.view(256, 39)
                    # loss = loss_function(pred, labels)

                    if uncertainty:
                        pred = model(pred, variance)
                    else:
                        pred = model(pred)
                    pred = pred[:, -1, :].to(self.device)
                    
                    #!transformer
                    # pred2 = transformer(pred, variance)
                    # pred2 = transformer(pred)
                    # pred2 = pred2.to(self.device)

                    end_time = time.time()
                    inference_time += end_time - start_time

                    # if uncertainty:
                    # weight_threshold = 0.5
                    # loss2 = loss_function(pred2, labels)
                    if uncertainty:
                        loss2 = loss_function_un(pred, labels, variance)
                    else:
                        loss2 = loss_function(pred, labels)
                    # loss = loss_function_un(pred, labels, variance)
                    # if loss < 0:
                    #     test_loss_error_counter += 1
                    test_loss += loss2.item()
                    # loss += weight * torch.mean(variance ** 2)
                    # performance = compute_performance(pred, labels)
                    # optimizer2.zero_grad()
                    # loss2.backward()
                    # optimizer2.step()
                        

                        # file.write(f'epoch: {epoch} i: {i} uncertainty: {variance.mean()}')
                        # if variance.mean() > weight_threshold and performance < 0.9:
                        #     weight *= weight_decay_rate
                        # elif performance < 0.0001:
                        #     weight /= weight_decay_rate
                    # else:
                    #     loss = loss_function(pred, labels)
                    #     test_loss += loss.item()


                    # loss = loss_function(pred, labels)
                    # test_loss += loss.item()

                    test_mae[i * self.batch_size:(i + 1) * self.batch_size, :] = get_mae(pred, labels)
                    test_distance[i * self.batch_size:(i + 1) * self.batch_size, :] = get_distance(pred, labels)
                    # test_mae2[i * self.batch_size:(i + 1) * self.batch_size, :] = get_mae(pred2, labels)
                    # test_distance2[i * self.batch_size:(i + 1) * self.batch_size, :] = get_distance(pred2, labels)

            inference_time /= len(self.test_loader)
            train_loss = train_loss * 1000 / len(self.train_loader)
            test_loss = test_loss * 1000 / len(self.test_loader)

            train_mae_mean = train_mae.mean(0) * 100
            test_mae_mean = test_mae.mean(0) * 100
            # train_mae_std = train_mae.std(0) * 100
            # test_mae_std = test_mae.std(0) * 100

            train_mae_avg = train_mae_mean.mean().item()
            test_mae_avg = test_mae_mean.mean().item()

            train_distance_mean = train_distance.mean(0) * 100
            test_distance_mean = test_distance.mean(0) * 100
            # train_distance_std = train_distance.std(0) * 100
            test_distance_std = test_distance.std(0) * 100

            train_distance_avg = train_distance_mean.mean().item()
            test_distance_avg = test_distance_mean.mean().item()

            #!transformer test distance
            # test_mae_mean2 = test_mae2.mean(0) * 100
            # test_mae_avg2 = test_mae_mean2.mean().item()
            # test_distance_mean2 = test_distance2.mean(0) * 100
            # test_distance_std2 = test_distance2.std(0) * 100
            # test_distance_avg2 = test_distance_mean2.mean().item()

            print(f'[{datetime.now()}] Epoch {epoch} evaluation report: ')
            print(f'| Train time:\t{train_time:.3f} sec')
            print(f'| Inference:\t{inference_time} sec')
            # print(f'| Train loss:\t{train_loss}')
            # print(f'| Test  loss:\t{test_loss}')
            # print(f'| Train MAE :\t{train_mae_avg} cm')
            # print(f'| Test  MAE :\t{test_mae_avg} cm')
            print(f'| Train dist:\t{train_distance_avg} cm')
            print(f'| Test  dist:\t{test_distance_avg} cm')

            if lr_decayer:
                lr_decayer.step()
            if lr_decayer2:
                lr_decayer2.step()

            result_header = ['train_loss', 'test_loss', 'train_mae', 'test_mae', 'train_distance', 'test_distance_mean', 'test_distance_std', 'nose_mean',
                             'left_shoulder_mean', 'right_shoulder_mean', 'left_elbow_mean', 'right_elbow_mean', 'left_wrist_mean', 'right_wrist_mean',
                             'left_pinky_mean', 'right_pinky_mean', 'left_index_mean', 'right_index_mean', 'left_thumb_mean', 'right_thumb_mean', 'nose_std',
                             'left_shoulder_std', 'right_shoulder_std', 'left_elbow_std', 'right_elbow_std', 'left_wrist_std', 'right_wrist_std',
                             'left_pinky_std', 'right_pinky_std', 'left_index_std', 'right_index_std', 'left_thumb_std', 'right_thumb_std']

            result = [train_loss, test_loss, train_mae_avg, test_mae_avg, train_distance_avg, test_distance_avg, test_distance_mean.std().item()]
            result.extend(test_distance_mean.tolist())
            result.extend(test_distance_std.tolist())

            df = pd.DataFrame([result], columns=result_header, index=[epoch])
            if epoch == 1:
                df.to_csv(self.result_path / 'result.csv', mode='a+')
            else:
                df.to_csv(self.result_path / 'result.csv', mode='a+', header=None)


            if test_distance_avg <= min_test_distance_avg:
                min_test_distance_avg = test_distance_avg
                torch.save(model.state_dict(), self.result_path / f'checkpoints/{epoch}.pt')

            # if uncertainty:
            #     log_header = ['Transformer test distance mean', 'Transformer test distance joint std', 'Transformer test distance joint mean']
            #     # tmp = variance.tolist()
            #     # for i, t in enumerate(tmp):
            #     #     tmp[i] = f'{t:.10f}'
            #     # log = [f'{test_distance_avg2}', f'{test_distance_std2}', f'{test_distance_mean2}']
            #     # log_header.extend(['uncertainty', 'max uncertainty', 'min uncertainty'])
            #     # log.extend([f'{variance.mean().item():.10f}', f'{variance.max().item():.10f}', f'{variance.min().item():.10f}'])
                
            #     df = pd.DataFrame([log], columns=log_header, index=[epoch])
            #     if epoch == 1:
            #         df.to_csv(self.result_path / 'log.csv', mode='a+')
            #     else:
            #         df.to_csv(self.result_path / 'log.csv', mode='a+', header=None)