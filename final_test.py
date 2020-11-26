from final_func import * 
from final_cons import *
from final_preprocess import * 
import torch.optim as optim
import torch.nn.functional as F
import sys

if EMBEDDINGALG == 'CBOW':
    w2v_path = './model/w2v_CBOW.model'
else:
    w2v_path = './model/w2v_skipgram.model'
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs>=0.5] = 1 # 大於等於0.5為災難
            outputs[outputs<0.5] = 0 # 小於 0.5為非災難
            ret_output += outputs.int().tolist()
    return ret_output

if __name__ == '__main__':
    test_path = sys.argv[1]
    predict_path = sys.argv[2]
    print("loading testing data ...")
    test_x,test_id = load_data(test_path, test=True)
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()
    test_dataset = TwitterDataset(X=test_x, y=None)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = 8)
    print('\nload model ...')
    if EMBEDDINGALG == 'CBOW':
        model = torch.load('./model/RNN_CBOW_best.model', map_location=device)
    else:
        model = torch.load('./model/RNN_skipgram_best.model', map_location=device)
    outputs = testing(batch_size, test_loader, model, device)
    tmp = pd.DataFrame({"id":[str(i) for i in test_id],"target":outputs})
    print("save csv ...")
    tmp.to_csv(predict_path, index=False)
    print("Finish Predicting")
