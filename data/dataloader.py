# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 16:21:21
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 11:47:28
import torch
import torch.utils.data as Data
from data.readdata import DataReaderseq as DataReader
import numpy as np

def getDataLoader(batch_size, dataset, num_of_questions, max_step,sourceroot_path,sourcedatadir_path,sourcedata_path,qm_data_path,testseqlen):
    if dataset == 'assist2015':
        handle = DataReader('./csv/assist2015/2015_builder_train.csv',
                            './csv/assist2015/2015_builder_test.csv', 
                            max_step,
                            100,testseqlen)
        real_qm_path = "./csv/assist2015/assist2015_q_matrix20_10_skills_steps.csv"
    elif dataset == 'assist2009updated':
        handle = DataReader('./csv/assist2009_updated/assist2009_updated_train.csv',
                            './csv/assist2009_updated/assist2009_updated_test.csv', 
                            max_step,
                            110,testseqlen)
        real_qm_path = "./csv/assist2009_updated/assist2009_train5_output30_30_skills_steps.csv"
    elif dataset == 'assist2017':
        handle = DataReader('./csv/assist2017/assist2017_train.csv',
                            './csv/assist2017/assist2017_test.csv', 
                            max_step,
                            102,testseqlen)
        real_qm_path = "./csv/assist2017/assist2017_q_matrix30_10_skills_steps.csv"     
    elif dataset == 'simu':
        handle = DataReader(sourceroot_path+sourcedatadir_path+sourcedata_path+'_train_deep.csv',
                            sourceroot_path+sourcedatadir_path+sourcedata_path+'_test_deep.csv', 
                            max_step,
                            num_of_questions,testseqlen)
        real_qm_path = sourceroot_path+sourcedatadir_path+qm_data_path    
    [trainData,testData]= handle.getAllData()
    
    dtrain = torch.tensor(trainData.astype(float).tolist(),
                          dtype=torch.float32)
    dtest = torch.tensor(testData.astype(float).tolist(),
                         dtype=torch.float32)
    trainLoader = Data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    testLoader = Data.DataLoader(dtest, batch_size=batch_size, shuffle=False)
    #读取q-matrix
    q_matrix = np.loadtxt(real_qm_path,delimiter=',')
    print("in dataloader q_matrix.shape:",q_matrix.shape)
    q_matrix = torch.from_numpy(q_matrix)
    
    q_matrix = q_matrix.float()
    #q_matrix.to(device)
    print("q_matrix.shape:",q_matrix.shape)
    return trainLoader, testLoader,q_matrix

