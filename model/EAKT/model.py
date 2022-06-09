import torch
import torch.nn as nn

from model.EAKT.attention import Encoder
from model.EAKT.embedding import Embedding

class EAKTModel(nn.Module):
    def __init__(self, heads, length, d_model, n_question, kc_dim, dropout , device, sigmoida, sigmoidb): 
        #              8,     50,     128,     124,        10,     0.1, 
        super(EAKTModel, self).__init__()
        self.embedding = Embedding(kc_dim, length, d_model)
        self.encoder = Encoder(heads, length, d_model, dropout)
        self.w = nn.Linear(d_model, n_question)
        self.sig = nn.Sigmoid()
        self.device = device
        self.kc_dim = kc_dim
        self.kc = nn.Linear(in_features=int(n_question),out_features=self.kc_dim)
        self.out = nn.Linear(in_features=self.kc_dim,out_features=1)
        self.hidden = nn.Linear(in_features=d_model,out_features=self.kc_dim)
        self.sigmoida = sigmoida
        self.sigmoidb = sigmoidb  
        
    def mySigmoid(self, x):
        return torch.div(torch.ones_like(x), torch.ones_like(x) + torch.exp(-torch.mul(x,self.sigmoida)-torch.ones_like(x)*self.sigmoidb))

    def forward(self, y ,device,flag):  # shape of input: [batch_size, length, questions * 2]
        batch = y 
        lenth_batch =int(int(batch.shape[2])/2)
#         print("lenth_batch",lenth_batch)
        y_kc =torch.cat((self.kc(y[:,:,0:lenth_batch]),self.kc(y[:,:,lenth_batch:])),2)
        x, y = self.embedding(y_kc ,device)  # shape: [batch_size, length, d_model]
        encode = self.encoder(x, y)  # shape: [batch_size, length, d_model]
        encode =encode.to(device)
        
        next_batch = batch.roll(shifts=(0, -1), dims=(0, 1))
        lenth_batch =int(int(next_batch.shape[2])/2)
        next_batch = next_batch[:,:,0:lenth_batch]+next_batch[:,:,lenth_batch:]
        next_kc = self.kc(next_batch)
        stu_state = self.hidden(encode)
#         stu_state = encode
        #version 2 
        cosine_similarity = torch.cosine_similarity(stu_state,next_kc,dim=2)
        stu_state_f2 = torch.norm(stu_state,dim=2)
        next_kc_hot_f2 = torch.norm(next_kc,dim=2)
        projection = torch.mul(stu_state_f2,cosine_similarity)
        subprojection = projection - next_kc_hot_f2 

        subprojection = torch.clamp(subprojection, -15, 15)
        res = self.mySigmoid(subprojection)       
#         res = self.sig(subprojection)  # shape of res: [batch_size, length, question]
        if flag :
            return res, stu_state
        else :
            return res, None  
