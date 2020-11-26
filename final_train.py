import pandas as pd 
import numpy as np 
from final_func import *
from final_cons import *
from final_preprocess import *
from torch import optim
import matplotlib.pyplot as plt
import sys
# detect device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load training and testing data and separate sentences into words
train_path = sys.argv[1]
test_path = sys.argv[2]
X_train, y_train = load_data(train_path)
X_test, _ = load_data(test_path, test=True)

#print(len(X_train), len(y_train),len(X_test))

# train a word embedding model or load a pre-trained word embedding model 
if EMBEDDING:
    model = train_word2vec(X_train+X_test)
    model.save('./model/w2v_{}.model'.format(EMBEDDINGALG))
    w2v_path = './model/w2v_{}.model'.format(EMBEDDINGALG)
else:
    if EMBEDDINGALG == 'skipgram':
        w2v_path = './model/w2v_skipgram.model'
    else:
        w2v_path = './model/w2v_CBOW.model'

# preprocess the words data list into the words embedding index data list
preprocess = Preprocess(X_train, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
X_train = preprocess.sentence_word2idx()
y_train = preprocess.labels_to_tensor(y_train)
#print(X_train.shape, X_train[0], y_train.shape, y_train[0])

# declare the LSTM model
model = LSTM_Net(embedding, embedding_dim=150, hidden_dim=100, num_layers=3, dropout=0.5, fix_embedding=fix_embedding).to(device)
# make validation set 
X_train_len = len(X_train)
X_train, X_val, y_train, y_val = X_train[:9*X_train_len//10], X_train[9*X_train_len//10:], y_train[:9*X_train_len//10], y_train[9*X_train_len//10:]
X_train, y_train = X_train[:], y_train[:]

# 把 data 做成 dataset 供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把 data 轉成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True,num_workers = 0)
val_loader = torch.utils.data.DataLoader(dataset = val_dataset,batch_size = batch_size,shuffle = False,num_workers = 0)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
model.train() 
criterion = nn.BCELoss() 
t_batch = len(train_loader) 
v_batch = len(val_loader) 
optimizer = optim.Adam(model.parameters(), lr=lr) 
total_loss, total_acc, best_acc = 0, 0, 0

# Drawing training curve
train_loss = []
train_acc = []
val_loss = []
val_acc = []

for epoch in range(epoch):
    total_loss, total_acc = 0, 0
    # 這段做 training
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device, dtype=torch.long) 
        labels = labels.to(device, dtype=torch.float) 
        optimizer.zero_grad() # 由於 loss.backward() 的 gradient 會累加，所以每次餵完一個 batch 後需要歸零
        outputs = model(inputs) # 將 input 餵給模型
        outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
        loss = criterion(outputs, labels) # 計算此時模型的 training loss
        loss.backward() # 算 loss 的 gradient
        optimizer.step() # 更新訓練模型的參數
        correct = evaluation(outputs, labels) # 計算此時模型的 training accuracy
        total_acc += (correct / batch_size)
        total_loss += loss.item()
    print('Epoch{}: \nTrain | Loss:{:.5f} Acc: {:.3f}'.format(epoch+1,total_loss/t_batch, total_acc/t_batch*100))
    train_loss.append(total_loss/t_batch) 
    train_acc.append(total_acc/t_batch*100)
    # 這段做 validation
    model.eval() # 將 model 的模式設為 eval，這樣 model 的參數就會固定住
    with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device, dtype=torch.long) 
            labels = labels.to(device, dtype=torch.float) 
            outputs = model(inputs) # 將 input 餵給模型
            outputs = outputs.squeeze() # 去掉最外面的 dimension，好讓 outputs 可以餵進 criterion()
            loss = criterion(outputs, labels) # 計算此時模型的 validation loss
            correct = evaluation(outputs, labels) # 計算此時模型的 validation accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
        print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
        val_loss.append(total_loss/v_batch)
        val_acc.append(total_acc/v_batch*100)
        if total_acc > best_acc:
            # 如果 validation 的結果優於之前所有的結果，就把當下的模型存下來以備之後做預測時使用
            best_acc = total_acc
            #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
            if EMBEDDINGALG == 'CBOW':
                torch.save(model, "./model/RNN_CBOW_best.model")
            else:
                torch.save(model, "./model/RNN_skipgram_best.model")
            print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
            print('-----------------------------------------------')
    model.train() # 將 model 的模式設為 train，這樣 optimizer 就可以更新 model 的參數（因為剛剛轉成 eval 模式）
# plot 
plt.plot(np.arange(1,epoch+2),train_loss)
plt.plot(np.arange(1,epoch+2), val_loss)
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train_loss','val_loss'])
plt.savefig('./figure/loss.png')
plt.clf()
plt.plot(np.arange(1,epoch+2),train_acc)
plt.plot(np.arange(1,epoch+2), val_acc)
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend(['train_acc','val_acc'])
plt.savefig('./figure/acc.png')
    