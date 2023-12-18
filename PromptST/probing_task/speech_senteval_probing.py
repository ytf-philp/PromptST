import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import  os
import torch.optim as optim
import numpy as np
from transformers import SpeechEncoderDecoderConfig, SpeechEncoderDecoderModel
from transformers import SpeechEncoderDecoderModel, Speech2Text2Processor
from transformers import modeling_outputs
from datasets import load_dataset
import torch.utils.data as Data
import soundfile as sf
import torch.nn as nn
from datasets import load_dataset
import pandas as pd
import os
from collections import Iterable
import functools

def compare(A,B):
    A=filter(str.isdigit,A)
    B=filter(str.isdigit,B)
    a=int("".join(list(A)))
    b=int("".join(list(B)))
    if a<b:
        return -1
    else:
        return 1


def get_model(model_path="facebook/s2t-wav2vec2-large-en-de",device="cuda:0"):
    configg=SpeechEncoderDecoderConfig.from_pretrained(model_path)
    processor = Speech2Text2Processor.from_pretrained(model_path)
    model = SpeechEncoderDecoderModel.from_pretrained(
    model_path,config=configg)
    model.to(device)
    decoder_input_ids = torch.tensor([[model.config.decoder.decoder_start_token_id]]).to(device)
    return [processor,model,decoder_input_ids]  #--->return list

def save_checkpoint(state_dict, file_name):
    torch.save(state_dict, file_name)

def save_csv(data,label,task,name,sche_step):
    roo="sentence_embedding/"+str(sche_step)+"/"
    save_folder=roo+task
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(save_folder+'/'+name):
        data=data.detach().numpy()
        label=label.detach().numpy()
        pdd=pd.DataFrame(data)
        pdd["label"]=label
        pdd.to_csv(save_folder+'/'+name)
    
class MDataSet(Dataset):
    def __init__(self, file_path, wav_path):
        self.split_num=["tr","te","va"]
        self.save_split=[] 
        self.file_path=file_path
        fulldata = open(file_path).readlines()
        fulldata = [line.split('\t') for line in fulldata]
        self.fulldata = pd.DataFrame(fulldata, columns=['split', 'cate', 'sent'])
        #self.sent_represents = open(sent_rep_path,'r').readlines()
        self.categories = self.fulldata.iloc[:,1].unique()
        self.fulldata['label']=0
        for i in range(len(self.categories)):
            self.fulldata['label'][self.fulldata['cate']==self.categories[i]] = i

        fs=os.listdir(wav_path)
        fs.sort(key=functools.cmp_to_key(compare))
        audio=[]
        for au in fs:
            audio.append(wav_path+'/'+au)
        a=pd.Series(audio)
        a=a.rename("file")
        self.unsplit_data=pd.concat([self.fulldata,a],axis=1)
        self.set_split() #根据tr、te、va进行分类
        self.get_dataset()
        pass

    #按照标注分好类
    def set_split(self):
        for split in self.split_num:
            data_tmp = self.unsplit_data[self.fulldata['split'] == split]
            if not os.path.exists(root+"/dataset_csv"):
                os.makedirs(root+"/dataset_csv")
            if(len(data_tmp))!= 0:
                self.save_split.append(split)
                data_tmp.to_csv(root+"/dataset_csv/senteval_"+split+".csv",index=None)

    def __len__(self):
        return len(self.data)
    def get_categories(self):
        return self.get_categories()

    def get_dataset(self):
        train_csv=root+"/dataset_csv/senteval_tr.csv"
        test_csv=root+"/dataset_csv/senteval_te.csv"
        validate_csv=root+"/dataset_csv/senteval_va.csv"
        data_file={}
        for split in self.save_split:
            if split == "tr":
                data_file["train"]=train_csv
            elif split == "te":
                data_file["test"]=test_csv
            elif split =="va":
                data_file["validate"]=validate_csv
        self.dataset=load_dataset('csv', data_files=data_file)
        
#输出csv转换后的dataset,device,batch
class DatasetFromCSV(Dataset):
    def __init__(self, csv_path):
 
        self.data = pd.read_csv(csv_path)
        self.sentence=torch.Tensor(np.array(self.data.iloc[:,:-1]))
        self.labels = torch.Tensor(np.array(self.data.iloc[:,-1]))
 
    def __getitem__(self):

        return (self.sentence, self.labels)
 
    def __len__(self):
        return len(self.data.index)

#这俩的batch仅预防显存溢出
class get_embedding():
    def __init__(self,add,datas,devcie,batch,model_list):
        #model loadding
        self.add=add
        self.data=datas
        self.device=devcie
        self.processor = model_list[0]
        self.model = model_list[1]
        self.decoder_input_ids = model_list[2]
        self.trans_wav()
        self.data_batch(batch)
        #print(self.databatch)
        self.features_new=[]
        for i in range(len(self.databatch)):
            self.features_new.append(torch.tensor([item.cpu().detach().numpy() for item in self.process_data(i,add)],device='cpu'))
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
    def process_data(self,i,add):
        features=[]
        #提取sentenc_embedding
        for audio_input in self.databatch[i]["audio"]:
            #print(audio_input)
            audio_input=torch.Tensor(audio_input)
            input_values = self.processor(audio_input,sampling_rate=16000 ,return_tensors="pt").input_values.to(self.device)
            outputs = self.model(input_values=input_values, decoder_input_ids=self.decoder_input_ids,output_hidden_states=True)
            torch.cuda.empty_cache(),torch.cuda.empty_cache(),torch.cuda.empty_cache(),torch.cuda.empty_cache(),torch.cuda.empty_cache()
            features.append(outputs.encoder_hidden_states[add]) 
            del outputs,audio_input
            torch.cuda.empty_cache()
        #通过求均值的方式转化成1*1024
        
        features_avg=[]
        for feature in features:
            mean = torch.mean(feature,1)
            std = torch.std(feature,1)
            stat_pooling = torch.cat((mean,std),1)
            features_avg.append(stat_pooling.squeeze())
        
        features_avg=torch.tensor([item.cpu().detach().numpy() for item in features_avg],device='cpu')
        del features
        torch.cuda.empty_cache()
        print("features_avg",features_avg.shape)
        return features_avg

    def result(self):
        self.features_cat=torch.cat(self.features_new,dim=0)

    def flatten(self, items, ignore_types=(str, bytes)):
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, ignore_types):
                yield from self.flatten(x)
            else:
                yield x
    def get_label(self):
        label=[]
        for i in range(len(self.databatch)):
            label.append(self.databatch[i]["label"])
        self.label=torch.tensor(list(self.flatten(label)))
        return self.label

    def get_item(self):
        self.get_label()
        return self.features_cat, self.label

    def trans_wav(self):
        audio=[]
        #train
        for i in range(len(self.data)):
            print(self.data["file"][i])
            speech,_=sf.read(self.data["file"][i])
            audio.append(speech)
        self.data=self.data.add_column("audio",audio)
        #print(self.data)
        return self.data

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

def getDataSplit(dataset: MDataSet,device,split,model_list,batch_size,task,layer_choice):
    datasets=dataset.dataset
    data,label=get_embedding(layer_choice,datasets[split],device,batch_size,model_list).get_item()
    save_csv(data,label,task,split+".csv",layer_choice)
    tensor_data=Data.TensorDataset(data,label)
    print(data.shape)

    train_load = torch.utils.data.DataLoader(tensor_data, batch_size, shuffle=True, num_workers=4)
    print('train data num:', len(train_load))
    return train_load

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
       
def train(task,dataset,dataloader,model_list, device,num_epochs=70, lr=0.001, save_path="", keep_epoch=10, resume=False,bs=1,layer_choice=0):
    class_num = len(dataset.categories)
    net = Classify(class_num)
    
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)

    #save model
    print(f"save_path={save_path}")
    if resume and os.path.exists(f"{save_path}/checkpoint_last.pt"):
        saved = torch.load(f"{save_path}/checkpoint_last.pt")
        start_epoch = saved['epoch']+1
        if 'model' in saved.keys():
            net.load_state_dict(saved['model'])
        else:
            net.load_state_dict(saved['state_dict'])
        print("zzzz-I start from epoch {start_epoch}")

    criterion = nn.CrossEntropyLoss()
    net.train()
    
    dataloader1 = getDataSplit(dataset,device,'validate',model_list,bs,task,layer_choice)
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


def train_validate(model_list,task,bs,device, num_epochs=70, lr=0.001,layer_choice=0):
    print(f">>> {task}")
    dataset = MDataSet(rf"{root}/data/probing_text/{task}.txt", rf"{root}/data/sent_audio/{task}")
    print("------------------------------------start train--------------------------------------------")
    
    save_folder = f"{root}/probing_generate/{task}/{str(layer_choice)}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    dataloader=getDataSplit(dataset,device,"train",model_list,bs,task,layer_choice)
    
    print("----------------------------------train feature extract finished-----------------------------------------")
    net, epoch = train(task=task,
                       dataset=dataset,
                       dataloader=dataloader,
                       model_list=model_list,
                       device=device,
                       save_path=save_folder,
                       num_epochs=num_epochs,
                       lr=lr,bs=bs,layer_choice=layer_choice)
    
    print("-----------------------------------finish train------------------------------")
    print("-----------------------------------start train-test-----------------------------")
    acc = validate(net, dataloader,device)
    print(f"zzzz-Z tr epoch={epoch} task={task} categories={len(dataset.categories)}, acc={acc}")

    print("--------------------------------------start test----------------------------------------")
    dataloader1 = getDataSplit(dataset,device, 'test',model_list,bs,task,layer_choice)
    acc = validate(net, dataloader1,device)
    print(f"zzzz-Z va epoch={epoch} task={task} categories={len(dataset.categories)}, acc={acc}")



if __name__ == '__main__':
    import sys
    import argparse
    # 创建解析器
    parser = argparse.ArgumentParser(description="Script Arguments")
    # 添加参数
    parser.add_argument('--root', default="." , help='Schedule Step')
    parser.add_argument('--layer_choice', type=int, default=12, help='Schedule Step')
    parser.add_argument('--cuda', type=str, default="0", help='CUDA device')
    parser.add_argument('--task', type=str, default="bigram_shift", help='Tasks')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model_path', type=str, default="/workspace/AmbigST-de/model_new", help='the model to analysis')
    
    args = parser.parse_args()

    layer_choice = args.layer_choice
    cuda = args.cuda
    task = args.task
    bs = args.bs
    num_epochs = args.num_epochs
    lr = args.lr
    root = args.root
    model_path = args.model_path
    device = torch.device("cuda:"+cuda if torch.cuda.is_available() else "cpu")
    model_list=get_model(model_path,device) #loading model

    print("--------------------------------------this is task:",task,"--------",layer_choice,"------------------------------------------------------------")
    train_validate(model_list,task, bs,device,num_epochs, lr,layer_choice)
