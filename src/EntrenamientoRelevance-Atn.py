#!/usr/bin/env python
# coding: utf-8

# # Entrenamiento (Clasificador Relevancia)

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from transformers import RobertaTokenizer, RobertaModel
from transformers import BertModel, BertTokenizer;
from torchsummary import summary
from torch.utils.data import DataLoader, random_split
import pandas as pd
from itertools import islice as take
from tqdm import trange

from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np
from os.path import join
import datetime
import time

import pandas as pd

import modelos as m


# In[2]:


# reproducibilidad
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[4]:


def timestamp(fmt='%y%m%dT%H%M%S'):
    """Regresa la marca de tiempo."""
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)


# ## Conjunto de Datos

# In[5]:


train_ds = m.ClaimsDS2I('relevance-DS_HNM.csv', nr=None)
print(len(train_ds))


# In[6]:


val_ds = m.ClaimsDS2I('relevance-DS_HNM_dev.csv', nr=None)
print(len(val_ds))


# In[7]:


#ids_, mask_,ids2_, mask2_, y = train_ds[:10]
#print(ids_.shape)
#print(mask_.shape)
#print(y)


# In[8]:


#train_size = int(0.8 * len(dataSet))
#val_size = len(dataSet) - train_size
#train_ds, val_ds = random_split(dataSet, [train_size, val_size])

#print(f'Train DS: {len(train_ds)}')
#print(f'Validation DS: {len(val_ds)}')


# ## Data Loader

# In[9]:


BATCH_SIZE = 64

train_dl = DataLoader(
    train_ds,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    # desordenar
    shuffle=True,
    # número de procesos paralelos
    num_workers=4,
    drop_last=True
)

val_dl = DataLoader(
    val_ds,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    # desordenar
    shuffle=True,
    # número de procesos paralelos
    num_workers=4
)


# In[10]:


#ids, masks,ids2, masks2, y = next(iter(train_dl))
#print(masks.shape)


# ## Modelo

# In[11]:


'''class RelevanceClassifier(nn.Module):
    def __init__(self):
        super(RelevanceClassifier, self).__init__()
        self.bert1 = BertModel.from_pretrained('bert-base-uncased')
        self.bert2 = BertModel.from_pretrained('bert-base-uncased')
        
        bert_size = 768
        
        self.interLayers = nn.Sequential(
            nn.Linear(bert_size*2, 2048),
            nn.Dropout(0.25),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, bert_size*2),
            nn.Dropout(0.25),
            nn.BatchNorm1d(bert_size*2),
            nn.Softmax(dim=1)
        )

        self.output = nn.Sequential(
                nn.Linear(bert_size*2, 2),
                nn.Softmax(dim = 1)
            )
        
    def evaluacion(self):
        for param in self.bert1.parameters():
            param.requires_grad = False
        for param in self.bert2.parameters():
            param.requires_grad = False
        self.eval()
        self.bert1.eval()
        self.bert2.eval()
        
    def entrenamiento(self):
        for param in self.bert1.parameters():
            param.requires_grad = True
        for param in self.bert2.parameters():
            param.requires_grad = True
        self.train()
        self.bert1.train()
        self.bert2.train()
    
    def forward(self, ids1, masks1, ids2, masks2):
        if ids1.type() == 'torch.cuda.FloatTensor':
            ids1=ids1.type(torch.LongTensor).to(device)
            masks1=masks1.type(torch.LongTensor).to(device)
            ids2=ids2.type(torch.LongTensor).to(device)
            masks2=masks2.type(torch.LongTensor).to(device)
        
        bert_output1 = self.bert1(input_ids=ids1, attention_mask=masks1, return_dict=True).last_hidden_state
        
        pool1 = bert_output1 * masks1.unsqueeze(2)
        pool1 = torch.div(pool1.sum(dim=1), masks1.sum(dim=1).unsqueeze(1))
        
        cls1 = bert_output1[:,0,:]
        
        bert_output2 = self.bert2(input_ids=ids2, attention_mask=masks2, return_dict=True).last_hidden_state
        
        pool2 = bert_output2 * masks2.unsqueeze(2)
        pool2 = torch.div(pool2.sum(dim=1), masks2.sum(dim=1).unsqueeze(1))
        
        cls2 = bert_output2[:,0,:]
        
        concat = torch.cat((pool1,pool2), 1)
        
        inter = self.interLayers(concat)
        
        concatCLS = torch.cat((cls1,cls2), 1)
        
        mul = torch.mul(concatCLS, inter)

        y = self.output(mul)
        return y
        '''


# In[12]:


#model = m.RelevanceClassifierTR().to(device)


# In[13]:


#ids = ids.to(device)
#masks = masks.to(device)
#ids = ids2.to(device)
#masks = masks2.to(device)
3
#y_hat = model(ids, masks, ids, masks)
#y_hat.shape


# In[14]:


#y_pred = torch.tensor([1 if y_>0.5 else 0 for y_ in y_hat])
#print(y_pred)
#acc = (y == y_pred).type(torch.float).mean()
#print(acc)


# In[15]:


#summary(model, [(1, 128), (1, 128), (1, 128), (1, 128)])


# ## Entrenamiento

# In[16]:


accumulation_steps = 4
def train_epoch(dl, model, opt):
    """Entrena una época"""
    # entrenamiento de una época
    index = 0
    for batch in dl:
        index+=1
        
        # vaciamos los gradientes
        opt.zero_grad()
        
        model = model.to(device)
        
        ids, masks, ids2, masks2, y_true = batch
        #ids, masks, y_true = batch
        ids = ids.to(device)
        masks = masks.to(device)
        ids2 = ids2.to(device)
        masks2 = masks2.to(device)
        y_true = y_true.to(device)
        
        #y_hat = model(ids, masks)
        y_hat = model(ids, masks, ids2, masks2)
        #y_hat = torch.squeeze(y_hat)
        
        # computamos la pérdida
        loss = F.cross_entropy(y_hat, y_true)
        
        #y_pred = torch.tensor([1 if y>0.5 else 0 for y in y_hat]).to(device)
        y_pred = torch.argmax(y_hat, 1)
        
        acc = (y_true == y_pred).type(torch.float32).mean()
        
        if index % 100 == 0:
            print(f"{index*BATCH_SIZE}: {acc}")
        
        # retropropagamos
        loss.backward()
        # actualizamos parámetros
        opt.step()


# In[17]:


def eval_epoch(dl, model, num_batches=None):
    """Evalua una época"""
    # evitamos que se registren las operaciones 
    # en la gráfica de cómputo
    with torch.no_grad():
        
        losses, accs = [], []
        # validación de la época con num_batches
        # si num_batches==None, se usan todos los lotes
        index=0
        for batch in take(dl, num_batches):
            index+=1
            
            model = model.to(device)
            
            ids, masks,ids2, masks2, y_true = batch
            #ids, masks, y_true = batch
            
            ids = ids.to(device)
            masks = masks.to(device)
            ids2 = ids2.to(device)
            masks2 = masks2.to(device)
            y_true = y_true.to(device)
            
            # hacemos inferencia para obtener los logits
            #y_hat = model(ids, masks)
            y_hat = model(ids, masks, ids2, masks2)
            #y_hat = torch.squeeze(y_hat, dim=1)
            
            # computamos la pérdida
            loss = F.cross_entropy(y_hat, y_true)
            # computamos la exactitud
            #y_pred = torch.tensor([1 if y>0.5 else 0 for y in y_hat]).to(device)
            y_pred = torch.argmax(y_hat, 1)
            acc = (y_true == y_pred).type(torch.float32).mean()
            
            if index % 100 == 0:
                print(f"VAL {index*BATCH_SIZE}: {acc}")

            # guardamos históricos
            losses.append(loss.item())
            accs.append(acc.item())

        loss = np.mean(losses) * 100
        acc = np.mean(accs) * 100
        
        return loss, acc


# In[18]:


def save_check_point(model, epoch, run_dir):
    """Guarda un punto de control."""
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, 
        join(run_dir, 'weights.pth')
    )


# In[19]:


def train(model, trn_dl, val_dl,
          trn_writer, val_writer, epochs,
          trn_batches=None, val_batches=None, lr=2e-5):

    # optimizador
    opt = optim.Adam(model.parameters(), lr=lr)

    # ciclo de entrenamiento
    best_acc = 0
    for epoch in trange(epochs):
        model.entrenamiento()

        # entrenamos la época
        train_epoch(trn_dl, model, opt)

        model.evaluacion()
        # evaluamos la época en entrenamiento
        trn_loss, trn_acc = eval_epoch(trn_dl, model, trn_batches)
        # registramos trazas de TB 
        trn_writer.add_scalar('metrics/loss', trn_loss, epoch)
        trn_writer.add_scalar('metrics/acc', trn_acc, epoch)
        
        print(f"Train: acc {trn_acc} loss {trn_loss}")

        
        # evaluamos la época en validación
        val_loss, val_acc = eval_epoch(val_dl, model, val_batches)
        # registramos trazas de TB
        val_writer.add_scalar('metrics/loss', val_loss, epoch)
        val_writer.add_scalar('metrics/acc', val_acc, epoch)
        
        print(f"Val: acc {val_acc} loss {val_loss}")

        # si hay mejora guardar punto de control
        if val_acc > best_acc:
            best_acc = val_acc
            save_check_point(model, epoch, run_dir)


# ## Ejecución

# In[20]:


run_dir = join('runs', 'relevance', timestamp())
run_dir


# In[21]:


train_writer = SummaryWriter(join(run_dir, 'trn'))
val_writer = SummaryWriter(join(run_dir, 'val'))


# In[22]:


# lanzamos Tensorboard
#%load_ext tensorboard
#%tensorboard --logdir runs/relevance --host localhost


# In[23]:


model = m.RelevanceClassifierSiamAtnMulLSTMPairShared().to(device)
#ckpt = torch.load(os.path.join('..','trained','siam-atn', 'weights_relevanceClassifier.pth'), map_location=device)
#state_dict = ckpt['model_state_dict']
#model.load_state_dict(state_dict)
#model.evaluacion()
#model.congelaBert(True)
#model = model.to(device)


# In[ ]:

model.congelaBert(True)
train(model, train_dl, val_dl, train_writer, val_writer, 
      epochs=5, trn_batches=1000, lr=2e-5)


# 
