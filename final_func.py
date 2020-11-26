import pandas as pd 
import numpy as np 
import torch
from torch import nn 
from gensim.models import word2vec
from final_cons import *
from torch.utils import data

# load training and testing data
def load_data(data_path, test=False):
    datas = pd.read_csv(data_path).to_numpy()
    if not test:
        X_data = [data.strip('\n').split(' ') for data in datas[:,3]]
        y_data = datas[:,4]
        return X_data, y_data 
    X_data = [data.strip('\n').split(' ') for data in datas[:,3]]
    data_id = datas[:,0]
    return X_data,data_id


def train_word2vec(x):
    # Train word to vector embedding model 
    if EMBEDDINGALG == 'skipgram':
        model = word2vec.Word2Vec(x, size=150, window=5, min_count=3, iter=15, sg=1)
    else:
        model = word2vec.Word2Vec(x, size=150, window=5, min_count=3, iter=15, sg=0)
    return model

def evaluation(outputs, labels):
    # outputs => probability (float)
    # labels => labels
    outputs[outputs>=0.5] = 1 # 大於等於 0.5 為正面
    outputs[outputs<0.5] = 0 # 小於 0.5 為負面
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 製作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否將 embedding fix 住，如果 fix_embedding 為 False，在訓練過程中，embedding 也會跟著被訓練
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最後一層的 hidden state
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x
class TwitterDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)