#!/usr/bin/env python
# coding: utf-8

# # Entrenamiento (Clasificador Veracidad)

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from transformers import RobertaTokenizer, RobertaModel
from transformers import BertModel, BertTokenizer, BertConfig;
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


train_ds = m.ClaimsDS2I('fake-truth-DS_bal_train.csv')
print(len(train_ds))


# In[6]:


val_ds = m.ClaimsDS2I('fake-truth-DS_dev.csv')
print(len(val_ds))


# In[7]:


#ids_, mask_, y = train_ds[:10]
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


#ids, masks, y = next(iter(train_dl))
#print(masks.shape)


# ## Modelo

# In[11]:


#model = RelevanceClassifier().to(device)


# In[12]:


#ids = ids.to(device)
#masks = masks.to(device)
#y_hat = model(ids, masks)


# In[13]:


#y_pred = torch.tensor([1 if y_>0.5 else 0 for y_ in y_hat])
#print(y_pred)
#acc = (y == y_pred).type(torch.float).mean()
#print(acc)


# In[14]:


#summary(model, [torch.zeros(1, 512, dtype=torch.long), torch.zeros(1, 512, dtype=torch.long)], 
#        device='cpu')


# ## Entrenamiento

# In[15]:


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


# In[16]:


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


# In[17]:


def save_check_point(model, epoch, run_dir):
    """Guarda un punto de control."""
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, 
        join(run_dir, 'weights.pth')
    )


# In[18]:


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

# In[19]:


run_dir = join('runs', 'fakedetection', timestamp())
run_dir


# In[20]:


train_writer = SummaryWriter(join(run_dir, 'trn'))
val_writer = SummaryWriter(join(run_dir, 'val'))


# In[21]:


# lanzamos Tensorboard
#%load_ext tensorboard
#%tensorboard --logdir runs/relevance --host localhost


# In[22]:


model = m.FakeClassifierSiamAtnMulLSTMPairShared().to(device)
ckpt = torch.load(os.path.join('..','trained','siam-atn', 'weights_fakeDetection_mul_shared_LSTM_roberta_pretrained.pth'), map_location=device)
state_dict = ckpt['model_state_dict']
model.load_state_dict(state_dict)
model = model.to(device)

model.congelaBert(False)

# In[ ]:


train(model, train_dl, val_dl, train_writer, val_writer, 
      epochs=3, trn_batches=1000, lr=2e-5)


# #### 
