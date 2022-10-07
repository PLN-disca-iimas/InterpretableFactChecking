#!/usr/bin/env python
# coding: utf-8

# # Verificación - BaseLine

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sentence_transformers import SentenceTransformer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from tqdm import trange

from itertools import islice as take

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from os.path import join
import datetime
import time


# In[2]:


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[3]:


#from torch.multiprocessing import Pool, Process, set_start_method
#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[119]:


def timestamp(fmt='%y%m%dT%H%M%S'):
    """Regresa la marca de tiempo."""
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)


# In[4]:


def carga_embeddings():
    print("Inicia carga embeddings...")
    archivo_emb=open("word_emb/glove.6B.300d.txt","r")
    embeddings={}
    for linea in archivo_emb:
        items=linea.split(" ")
        array=np.array(items[1:], dtype=np.float32)
        
        embeddings[items[0]]=array
        
    archivo_emb.close()
    
    print("Finaliza carga embeddings [OK]")
    
    return embeddings


# In[5]:


word_emb = carga_embeddings()


# In[6]:


len(word_emb)


# In[128]:


def get_embedding_word(word):
    emb=word_emb.get(word, None)
    
    if emb is not None:
        emb = np.concatenate((emb, np.ones(768-len(emb))))
        
    return emb

def get_embedding_sentence(sentence, padding):
    items=sentence.split(" ")
    s_embeddings=[]
    size_emb=768
    for item in items:
        if len(s_embeddings)==padding:
            break;
        w_embedding=get_embedding_word(item.lower())
        
        if w_embedding is None:
            continue;
        
        size_emb=len(w_embedding)
        s_embeddings.append(w_embedding)
    
    while padding>0 and len(s_embeddings)<padding:
        w_embedding=np.zeros(size_emb)
        s_embeddings.append(w_embedding)
    
    return s_embeddings


# In[8]:


sent_trans = SentenceTransformer('bert-base-nli-mean-tokens')


# In[38]:


class ClaimsDS:
    
    def __init__(self, file, nr=None):
        df = pd.read_csv(file, header=None)
        df = df.sample(frac=1).reset_index(drop=True)
        if nr is not None:
            df=df.sample(n=nr)
        
        self.Y = torch.tensor(df[0].to_numpy(), dtype=torch.long)
        self.X_1 = df[1].values.tolist()
        self.X_2 = df[2].values.tolist()
        
        self.X_1 = sent_trans.encode(self.X_1)
        
    def __getitem__(self, i):
        vect_claim = self.X_1[i]
        vect_evidence = get_embedding_sentence(self.X_2[i], 200)
        
        return torch.tensor(vect_claim, dtype=torch.float), torch.tensor(vect_evidence, dtype=torch.float), self.Y[i]
        
        
    def __len__(self):
        return len(self.Y)


# In[39]:


ds_train = ClaimsDS('fake-truth-DS-bl-bal_train.csv')
print(f'Train: {len(ds_train)}')

ds_val = ClaimsDS('fake-truth-DS-bl_dev.csv')
print(f'Train: {len(ds_val)}')


# In[11]:


item_0 = ds_train[1]


# In[12]:


item_0[2].shape


# In[131]:


BATCH_SIZE = 32

train_dl = DataLoader(
    ds_train,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    # desordenar
    shuffle=True,
    # número de procesos paralelos
    num_workers=3,
    drop_last=True
)

val_dl = DataLoader(
    ds_val,
    # tamaño del lote
    batch_size=BATCH_SIZE,
    # desordenar
    shuffle=True,
    # número de procesos paralelos
    num_workers=3
)


# In[41]:


for claim, evid, Y in train_dl:
    print(claim.shape)
    print(evid.shape)
    print(Y.shape)
    break


# ### Modelo

# In[109]:


class FakeClassifierLSTM(nn.Module):
    def __init__(self):
        super(FakeClassifierLSTM, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(1536, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )
        
        self.lstm = nn.LSTM(input_size=768, batch_first=True, bidirectional=True, hidden_size=384)
        
        self.dense2 = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
    def forward(self, claim_, evidence_):
        claim_ = claim_.unsqueeze(1).repeat(1, evidence_.shape[1], 1)
        
        in_dense1 = torch.cat([claim_, evidence_], dim=2)
        
        out_dense1 = self.dense1(in_dense1.float()).squeeze(2)
        
        out_dense1 = out_dense1.unsqueeze(1)
        
        out_lstm, _ = self.lstm(evidence_.float())
        
        atn_lstm = out_dense1 @ out_lstm
        atn_lstm = atn_lstm.squeeze(1)
        
        out_dense2 = self.dense2(atn_lstm)
        
        return out_dense2


# In[113]:


def train_epoch(dl, model, opt):
    """Entrena una época"""
    # entrenamiento de una época
    index = 0
    for batch in dl:
        index+=1
        
        # vaciamos los gradientes
        opt.zero_grad()
        
        model = model.to(device)
        
        claim, evid, y_true = batch
        
        claim = claim.to(device)
        evid = evid.to(device)

        y_true = y_true.to(device)
        
        #y_hat = model(ids, masks)
        y_hat = model(claim, evid)
        
        # computamos la pérdida
        loss = F.cross_entropy(y_hat, y_true)
        
        y_pred = torch.argmax(y_hat, 1)
        
        acc = (y_true == y_pred).type(torch.float32).mean()
        
        if index % 100 == 0:
            print(f"{index*BATCH_SIZE}: {acc}")
        
        # retropropagamos
        loss.backward()
        # actualizamos parámetros
        opt.step()


# In[114]:


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
            
            claim, evid, y_true = batch
        
            claim = claim.to(device)
            evid = evid.to(device)

            y_true = y_true.to(device)
            
            # hacemos inferencia para obtener los logits
            y_hat = model(claim, evid)
            
            # computamos la pérdida
            loss = F.cross_entropy(y_hat, y_true)
            # computamos la exactitud
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


# In[116]:


def save_check_point(model, epoch, run_dir):
    """Guarda un punto de control."""
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, 
        join(run_dir, 'weights.pth')
    )


# In[117]:

best_acc = 0
def train(model, trn_dl, val_dl, epochs,
          trn_batches=None, val_batches=None, lr=1e-3):
    global best_acc
    # optimizador
    opt = optim.Adam(model.parameters(), lr=lr)

    # ciclo de entrenamiento
    for epoch in trange(epochs):
        model.train()

        # entrenamos la época
        train_epoch(trn_dl, model, opt)

        model.eval()
        # evaluamos la época en entrenamiento
        trn_loss, trn_acc = eval_epoch(trn_dl, model, trn_batches)
        
        print(f"Train: acc {trn_acc} loss {trn_loss}")

        # evaluamos la época en validación
        val_loss, val_acc = eval_epoch(val_dl, model, val_batches)
        
        print(f"Val: acc {val_acc} loss {val_loss}")

        # si hay mejora guardar punto de control
        if val_acc > best_acc:
            best_acc = val_acc
            save_check_point(model, epoch, run_dir)


# In[121]:


run_dir = join('runs', 'fakedetection', timestamp())
train_writer = SummaryWriter(join(run_dir, 'trn'))
val_writer = SummaryWriter(join(run_dir, 'val'))
run_dir


# In[133]:


model = FakeClassifierLSTM().to(device)


# In[134]:


train(model, train_dl, val_dl, 
      epochs=3, trn_batches=1000, lr=1e-3)
train(model, train_dl, val_dl, 
      epochs=2, trn_batches=1000, lr=1e-4)
train(model, train_dl, val_dl, 
      epochs=5, trn_batches=1000, lr=2e-5)


# In[ ]:




