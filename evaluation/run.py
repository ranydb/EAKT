# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-09 21:50:46
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 13:20:09
"""
Usage:
    run.py (eakt) --data=<h> --questions=<h> [options]

Options:
    --length=<int>                      max length of question sequence [default: 50]
    --questions=<int>                   num of question [default: 124]
    --lr=<float>                        learning rate [default: 0.001]
    --bs=<int>                          batch size [default: 64]
    --seed=<int>                        random seed [default: 59]
    --epochs=<int>                      number of epochs [default: 30]
    --cuda=<int>                        use GPU id [default: 0]
    --hidden=<int>                      dimention of hidden state [default: 128]
    --kc=<int>                          knowledge compoments dimention [default: 10]
    --layers=<int>                      layers of rnn or transformer [default: 1]
    --heads=<int>                       head number of transformer [default: 8]
    --dropout=<float>                   dropout rate [default: 0.1]
    --beta=<float>                      reduce rate of MyModel [default: 0.95]
    --data=<string>                     dataset [default: assist2009]
    --kernels=<int>                     the kernel size of CNN [default: 7]
    --memory_size=<int>                 memory size of DKVMN model [default: 20]
    --weight_decay=<float>              weight_decay of optimizer [default: 0]
    --sigmoida=<float>                  coefficient of custom sigmoid function [default: 5]
    --sigmoidb=<float>                  constant custom sigmoid function [default: 6.9]
    --save_model=<bool>                 whether save the KT model [default: true]
    --save_epoch=<int>                  the epoch to save the KT model [default: 0]
    --gpu=<int>                         select the gpu card [default: 0]
"""

import os
import random
import logging
import torch
import sys
import torch.optim as optim
import numpy as np

from datetime import datetime
from docopt import docopt
from data.dataloader import getDataLoader
from evaluation import eval

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    args = docopt(__doc__)
    length = int(args['--length'])
    questions = int(args['--questions'])
    lr = float(args['--lr'])
    bs = int(args['--bs'])
    seed = int(args['--seed'])
    epochs = int(args['--epochs'])
    cuda = args['--cuda']
    hidden = int(args['--hidden'])
    kc = int(args['--kc'])
    layers = int(args['--layers'])
    heads = int(args['--heads'])
    dropout = float(args['--dropout'])
    beta = float(args['--beta'])
    dataset = str(args['--data'])
    kernel_size = int(args['--kernels'])
    memory_size = int(args['--memory_size'])
    weight_decay = float(args['--weight_decay'])
    sigmoida = float(args['--sigmoida'])
    sigmoidb = float(args['--sigmoidb'])
    save_model = bool(args['--save_model'])
    save_epoch = int(args['--save_epoch'])
    gpu = int(args['--gpu'])
    testseqlen=10
    model_type = 'EAKT'

    simu_root_folder="./csv/"
    simu_datadir_name ="simu/"
    simu_datafile_name="s20000_q30_kc10" 
    qm_path = "s20000_q30_kc10_seqlen50_groundtruth_qm.csv"

    logger = logging.getLogger('main')
    logger.setLevel(level=logging.DEBUG)
    date = datetime.now()
    if not os.path.exists(simu_root_folder+simu_datadir_name+'results'):
        os.makedirs(simu_root_folder+simu_datadir_name+'results')
    handler = logging.FileHandler(
        f'{simu_root_folder}{simu_datadir_name}results/{date.year}_{date.month}_{date.day}_{model_type}_result.log')
#     result_data=f'{simu_root_folder}{simu_datadir_name}results/{date.year}_{date.month}_{date.day}_{date.hour}:{date.minute}_{model_type}_result'
    result_data=f'{simu_root_folder}{simu_datadir_name}results/{date.hour}:{date.minute}:{date.second}_{model_type}_heads:{heads}_hidden:{hidden}_save_epoch:{save_epoch}_sigmoda:{sigmoida}_sigmodb:{sigmoidb}_result'
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info(' '.join(sys.argv))
    logger.info(list(args.items()))
    setup_seed(seed)

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        device = torch.device('cuda:'+str(gpu))
    else:
        device = torch.device('cpu')
    print("length:",length)

    trainLoader, testLoader,q_matrix = getDataLoader(bs, dataset, questions, length, simu_root_folder, simu_datadir_name, simu_datafile_name, qm_path, testseqlen)
    from model.EAKT.model import EAKTModel
    model = EAKTModel(heads, length, hidden, questions, kc,dropout , device, sigmoida, sigmoidb)
#   model.kc.weight = torch.nn.Parameter(q_matrix.t())
    model.kc.weight = torch.nn.Parameter(q_matrix.t(),requires_grad=False)
    model.kc.requires_grad = False
    print(model.kc.weight)

    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    loss_func = eval.lossFunc(questions, length, model_type ,device)
    model = model.to(device)
    save_stu_state = None
    final_auc = 0
    for epoch in range(epochs):
        print('epoch: ' + str(epoch))
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer,
                                          loss_func ,device)
        logger.info(f'epoch {epoch}')
        #stu_state=eval.test_epoch(model, testLoader, loss_func, device)
        
        stu_state_1,auc = eval.test_epoch_seq(model, testLoader, loss_func, device,testseqlen)
        if save_stu_state is None:
            save_stu_state = stu_state_1
            final_auc = auc
        else:
            if auc>final_auc:
                save_stu_state = stu_state_1
                final_auc = auc
        
        if save_model and save_epoch == epoch:
            print(stu_state_1.shape)
            #np.savetxt(result_data,stu_state.cpu().detach().numpy(), delimiter=',')
            np.save(result_data,save_stu_state)
            if weight_decay != 0:
                model_type = model_type + '+'
            torch.save(model, f"results/{date.year}_{date.month}_{date.day}_{date.hour}:{date.minute}_{model_type}_{dataset}_model.pt")
if __name__ == '__main__':
    main()
