 # -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 13:42:11
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:33:06
import tqdm
import torch
import logging
import numpy as np
import torch.nn as nn
from sklearn import metrics

logger = logging.getLogger('main.eval')


def performance(ground_truth, prediction,loss):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)

    f1 = metrics.f1_score(ground_truth.detach().cpu().numpy(),
                          torch.round(prediction).detach().cpu().numpy())
    recall = metrics.recall_score(ground_truth.detach().cpu().numpy(),
                                  torch.round(prediction).detach().cpu().numpy())
    precision = metrics.precision_score(
        ground_truth.detach().cpu().numpy(),
        torch.round(prediction).detach().cpu().numpy())
    logger.info('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' +
                str(recall) + ' precision: ' + str(precision)+ ' loss: '+str(loss))
    print('auc: ' + str(auc) + ' f1: ' + str(f1) + ' recall: ' + str(recall) +
          ' precision: ' + str(precision)+ ' loss: '+str(loss))
    return auc


class lossFunc(nn.Module):
    def __init__(self, num_of_questions, max_step,model_type,device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.num_of_questions = num_of_questions
        self.max_step = max_step
        self.device = device
        self.model = model_type

    def forward(self, pred, batch,isTest,testseqlen):
        loss = 0

        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        for student in range(pred.shape[0]):
            if self.model == 'ESF' or self.model == 'EAKT':
                p = pred[student][:-1]
                #p = p.squeeze(1)
            else:
                delta = batch[student][:, 0:self.num_of_questions] + batch[student][:, self.num_of_questions:]  # shape: [length, questions] 提取student的
                temp = pred[student][:self.max_step - 1].mm(delta[1:].t())
                index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                     dtype=torch.long, device=self.device)
                p = temp.gather(0, index)[0]

            
            flag = batch[student].sum(1)[1:]

            flag=flag.type(torch.long)
            a = (((batch[student][:, 0:self.num_of_questions] - 
                   batch[student][:, self.num_of_questions:]).sum(1) + 1) //
                 2)[1:]
            
            flaglen = False
            for i in range(len(flag) - 1, -1, -1):
                if flag[i] > 0 or i==0:
                    #flaglen = True
                    p = p[:i + 1]
                    a = a[:i + 1]
                    break
            #if(flaglen):
            #print("p:",p)
            #print("a:",a)
            if isTest:
                if len(p)>testseqlen:
                    p=p[-testseqlen:]
                    a=a[-testseqlen:]
            loss += self.crossEntropy(p, a)
            prediction = torch.cat([prediction, p])
            ground_truth = torch.cat([ground_truth, a])
        #print("tmp_list: \n",tmp_list)        
        return loss, prediction, ground_truth


def train_epoch(model, trainLoader, optimizer, loss_func ,device):
    model.to(device)
    
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        pred, tmp = model(batch , device, False)
          
        loss, prediction, ground_truth = loss_func(pred, batch,False,0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # print(f"train loss: {loss.item()/50:.4f}")
    # performance(ground_truth, prediction)
    #print(model.kc.weight) 
    return model, optimizer

def test_epoch_seq(model, testLoader, loss_func, device,testseqlen):
    model.to(device)
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    fullstu_state = None 
    # for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
    for i, batch in enumerate(testLoader):
        batch = batch.to(device)
        pred, stu_state = model(batch  ,device,True)
        data = []
        label = []
        loss, p, a = loss_func(pred, batch,True,testseqlen)
        prediction = torch.cat([prediction, p])
        #print("prediction:",prediction.shape)
        ground_truth = torch.cat([ground_truth, a])
        tmp_state=stu_state.cpu().detach().numpy()
        if(fullstu_state is None):
            fullstu_state = tmp_state
        else:
            fullstu_state=np.concatenate((fullstu_state,tmp_state),axis=0)
        #fullstu_state = torch.cat([fullstu_state, stu_state])
    # print(f"test loss: {loss.item()/50:.4f}")
    print("shape_ground:",ground_truth.shape)
    print("shape_prediction:",prediction.shape)
    #print(fullstu_state.shape,"fullstu_state")
    auc=performance(ground_truth, prediction,loss.item())
    
    return fullstu_state,auc
