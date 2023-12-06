import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json, os, shutil
import torch.optim as optim
from datasets import load_from_disk
import sys
import numpy as np
from transformers import BertConfig, Wav2Vec2Config, SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel
from transformers import SpeechEncoderDecoderModel, Speech2Text2Processor
from transformers import modeling_outputs
from datasets import load_dataset
import torch.utils.data as Data
import soundfile as sf
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import datetime
from datasets import load_dataset
#import plotly.io as pio
#pio.renderers.default = 'iframe_connected'
import csv
import pandas as pd
import os
from tqdm import tqdm
from collections import Iterable
import functools


def get_model(device):
    configg=SpeechEncoderDecoderConfig.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
    model = SpeechEncoderDecoderModel.from_pretrained(
    "facebook/s2t-wav2vec2-large-en-de",config=configg)
    
    model.to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(device)
    return [model,decoder_input_ids]  #--->return list

def save_csv(datas,labels,task,name,add):
    for ad in add:
        roo="/workspace/yutengfei6/users/yutengfei6/docker-remote/ASR_probing/"+str(ad)+"/"
        save_folder=roo+task
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(save_folder+'/'+name):
            data=datas[ad].numpy()
            label=labels.detach().numpy()
            pdd=pd.DataFrame(data)
            pdd["labels"]=label
            pdd.to_csv(save_folder+'/'+name)
#这俩的batch仅预防显存溢出
class get_embedding():
    def __init__(self,add,datas,devcie,batch,model_list):
        #model loadding

        self.data=datas[0:32170]
        print(len(self.data))
        self.features_cat=dict()
        #注意add是个列表
        self.add=add
        self.device=devcie
        self.model = model_list[0]
        self.decoder_input_ids = model_list[1]
        self.features_new_dic=dict()
        for ad in add:
            self.features_new_dic[ad]=[]
        for i in range(len(self.data["labels"])):
            result=self.process_data(i,add)
            for ad in add:
                self.features_new_dic[ad].append(torch.tensor(result[ad].cpu().detach().numpy(),device='cpu'))
                torch.cuda.empty_cache()

        self.result()
        del self.model
        torch.cuda.empty_cache()


    #将数据集分成几小批
    def data_batch(self,batch):
        self.databatch=[]
        for i in range(int(len(self.data)/batch)):
            if i*batch+batch <=len(self.data):
                self.databatch.append(self.data[i*batch:i*batch+batch])
            else:
                self.databatch.append(self.data[i*batch:])
        return self.databatch


    #分批次提取表征，并将结果保存在cpu中
    #得到一组（add 层）数据
    def process_data(self,i,add):
        features_avg=dict()
        for ad in add:
            features_avg[ad]=[]
        #提取sentenc_embedding
        input_value=torch.Tensor(np.array(self.data["inputs"][i])).to(self.device)
        attention_mask=torch.Tensor(self.data["attention_mask"][i]).to(self.device)
        outputs = self.model(input_values=input_value,attention_mask=attention_mask, decoder_input_ids=self.decoder_input_ids,output_hidden_states=True)
        del input_value,attention_mask
        torch.cuda.empty_cache()
        for ad in add:
            feature=outputs.encoder_hidden_states[ad]
            mean = torch.mean(feature,1)
            std = torch.std(feature,1)
            stat_pooling = torch.cat((mean,std),1)
            features_avg[ad]=stat_pooling
        del outputs
        torch.cuda.empty_cache()
        print("features_avg",features_avg[add[0]].shape)
        return features_avg


    def flatten(self, items, ignore_types=(str, bytes)):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, ignore_types):
                yield from self.flatten(x)
            else:
                yield x
    def get_label(self):
        label=[]
        for la in self.data["labels"]:
            label.append(int(la))
        self.label=torch.tensor(list(self.flatten(label)))
        return self.label
    def result(self):
        #对维度进行整合
        for ad in self.add:
            self.features_cat[ad]=torch.cat(self.features_new_dic[ad],dim=0)
    def get_item(self):
        self.get_label()
        return self.features_cat, self.label

        #将路径转换成wav


#保存成CSV文件以供dataloader提取

class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
 
        self.data = pd.read_csv(csv_path)
        self.sentence=self.data.iloc[:,0]
        self.labels = self.data.iloc[:, 1]
 
    def __getitem__(self):

        return (self.sentence, self.labels)
 
    def __len__(self):
        return len(self.data.index)

class Classify(nn.Module):
    def __init__(self, class_num):
        super(Classify, self).__init__()
        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 =nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0)

        self.fc2=nn.Linear(1024,1024)
        self.relu2 =nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(1024, 512)
        self.relu3 =nn.ReLU(inplace=True)
        self.drop3 = nn.Dropout(0.2)

        self.fc4=nn.Linear(512,class_num)
        # self.softmax = nn.Softmax()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.relu3(x)
        x = self.drop3(x)

        x = self.fc4(x)
        return x

def getDataSplit(datasets,device ,split,model_list,batch_size,task,sche_step,add):
    data,label=get_embedding(add,datasets[split],device,batch_size,model_list).get_item()
    #save_csv(data,label,task,split+".csv",add)
    #tensor_data=Data.TensorDataset(data,label)
    #print(data.shape)
    #train_load = torch.utils.data.DataLoader(tensor_data, batch_size, shuffle=True, num_workers=4)
    #print('train data num:', len(train_load))
    #return train_load

class AverageMeter():
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum /self.count
       
def train( task,dataset,dataloader,model_list, device,num_epochs=70, lr=0.001, sche_step=50, save_path="", keep_epoch=10, resume=False,bs=1,add=0):
    class_num = len(dataset.categories)
    net = Classify(class_num)
    
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sche_step, gamma=0.3)

    criterion = nn.CrossEntropyLoss()
    net.train()

    #将模型分批次转换成向量
    #dataloader=getDataSplit(dataset,device,"train",model_list)
    #print("----------------------------------train feature extract finished-----------------------------------------")
    dataloader1 = getDataSplit(dataset,device,'test',model_list,bs,task,sche_step,add)
    #dataloader1=dataloader
    print("----------------------------------validate feature extract finished-----------------------------------------")    
    losses=[]
    loss_meter = AverageMeter()
    for epoch in range(num_epochs):
        loss_meter = AverageMeter()
        for i, (inputs, labels) in enumerate(dataloader):
            #print(inputs.shape)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss_meter.update(loss.item(), 1)
            loss.backward()
            if i%10000 == 0:
                #print(outputs,labels)
                print(f"[{epoch}-{i}/{len(dataloader)}, lr={scheduler.get_lr()}] loss:", loss_meter.avg)
                #print(f"[{epoch}-{i}/{len(dataloader)}, lr={lr}] loss:", loss_meter.avg)

                losses.append((loss_meter.avg))
                loss_meter.reset()
            optimizer.step()


        #每10次训练测试一次
        if (epoch+1) % 10 ==0:
            print("----------------------------------validate test per 10 epoch-----------------------------------------")
            acc = validate(net, dataloader1, device)
            print(f"zzzz-Z va epoch={epoch} task={task} categories={len(dataset.categories)}, acc={acc}")
        

    return net, num_epochs

def validate(net, dataloader: DataLoader, device):
    accuracy = AverageMeter()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            output: torch.Tensor = net(inputs)
        output_result = output.argmax(dim=1)
        acc = (output_result == labels).float().mean()
        accuracy.update(acc, dataloader.batch_size)
    return accuracy.avg






def train_validate(model_list,task,bs,device, num_epochs=70, sche_step=50, lr=0.001,add=0):
    print(f">>> {task}")
    dataset =load_from_disk("/workspace/yutengfei6/users/yutengfei6/docker-remote/ASR_probing/data_new")

    print("------------------------------------start train--------------------------------------------")
    
    save_folder = f"{root}/probing_generate/{task}/{sche_step}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

#将模型分批次转换成向量
    #dataloader=getDataSplit(dataset,device,"train",model_list,bs,task,sche_step,add)
    #print("----------------------------------train feature extract finished-----------------------------------------")
    #net, epoch = train(task,dataset,dataloader,model_list,device ,save_path=save_folder, num_epochs=num_epochs, sche_step=sche_step, lr=lr,bs=bs,add=add)
    #print("-----------------------------------finish train------------------------------")

    #print("-----------------------------------start train-test-----------------------------")
    #acc = validate(net, dataloader,device)
    #print(f"zzzz-Z tr epoch={epoch} task={task} categories={len(dataset.categories)}, acc={acc}")

    print("--------------------------------------start test----------------------------------------")
    #dataloader = getDataSplit(dataset,device, 'train',model_list,bs,task,sche_step,add)
    dataloader1 = getDataSplit(dataset,device, 'test',model_list,bs,task,sche_step,add)
    #acc = validate(net, dataloader1,device)
    #print(f"zzzz-Z va epoch={epoch} task={task} categories={len(dataset.categories)}, acc={acc}")



if __name__ == '__main__':
    import sys
    #root = r"C:\Users\zhangzheng15\Desktop\notes\data_augmentation\datasets\SentEval\data"
    root = r"./SentEval-main"  #sent_represents/$feature/${ck_name}.${task}.1.txt"
    bs=1
    num_epochs = 10
    cudaa="7"
    #sche_step = [24]
    lr = 0.0001
    device = torch.device("cuda:"+cudaa if torch.cuda.is_available() else "cpu")
    print(f"-----------------------------------loading model ----------------------------------------")
    model_list=get_model(device)
    print(len(model_list))


    #nohup python -u senteval_fin.py > senteval.log 2>&1 &
    task = "phoneme"
    
    add=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    print("--------------------------------------this is task:",task,"--------")
    train_validate( model_list, task, bs,device,num_epochs, add , lr=lr,add=add)






        
        