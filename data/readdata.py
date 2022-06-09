# -*- coding: utf-8 -*-
# @Author: jarvis.zhang
# @Date:   2020-05-08 18:46:52
# @Last Modified by:   jarvis.zhang
# @Last Modified time: 2020-05-10 00:51:05
import numpy as np
import itertools
   
class DataReaderseq():
    def __init__(self, train_path, test_path, maxstep, numofques,testseqlen):
        self.train_path = train_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques
        self.testseqlen = testseqlen # 测试集长度
    def getData(self, file_path):
        data1 = []
        data2 = []
        maxlen = 0
        tmp_list = []
        total_questions = 0
        with open(file_path, 'r') as file:
            for leng, ques, ans in itertools.zip_longest(*[file] * 3):
                leng = int(leng.strip().strip(','))
                total_questions = total_questions+leng
                if leng >maxlen :
                    maxlen = leng
                ques = [int(q) for q in ques.strip().strip(',').split(',')]
                ans = [int(a) for a in ans.strip().strip(',').split(',')]
                slices = leng//self.maxstep + (1 if leng % self.maxstep > 0 else 0)
                tmp_list.append(leng)
                for i in range(slices):
                    temp1 = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                    temp2 = np.zeros(shape=[self.maxstep, 2 * self.numofques])
                    if leng > 0:
                        if leng >= self.maxstep:
                            steps = self.maxstep
                        else:
                            steps = leng
                        for j in range(steps):
                            if j<steps-self.testseqlen:
                                if ans[i*self.maxstep + j] == 1:
                                    temp1[j][ques[i*self.maxstep + j]-1] = 1
                                    temp2[j][ques[i*self.maxstep + j]-1] = 1
                                else:
                                    temp1[j][ques[i*self.maxstep + j] + self.numofques-1] = 1
                                    temp2[j][ques[i*self.maxstep + j] + self.numofques-1] = 1
                            else:
                                if ans[i*self.maxstep + j] == 1:
                                    temp2[j][ques[i*self.maxstep + j]-1] = 1
                                else:
                                    temp2[j][ques[i*self.maxstep + j] + self.numofques-1] = 1
                        leng = leng - self.maxstep
                    data1.append(temp1.tolist())
                    data2.append(temp2.tolist())
            print('done: ' + str(np.array(data1).shape))
            print('done: ' + str(np.array(data2).shape))
            print("max_step: ",maxlen)
            print("total_questions: ",total_questions)

        return data1,data2

    def getAllData(self):
        print('loading train data...')
        trainDataseq1,trainDataseq2 = self.getData(self.train_path)
        testDataseq1, testDataseq2= self.getData(self.test_path)
        finaltrainData=np.array(trainDataseq1+testDataseq1)
        finaltestData =np.array(trainDataseq2+testDataseq2)
        print(finaltrainData.shape)
        print(finaltestData.shape)
        return finaltrainData,finaltestData

