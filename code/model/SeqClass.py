import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path


class LinearClassifier(torch.nn.Module):
    def __init__(self, max_len : int, num_classes : int , model_path : str):
        super(LinearClassifier,self).__init__()
        model = BertModel.from_pretrained(model_path)
        self.ptmodel = model

        self.fctemp0 = torch.nn.Linear(768 * max_len,3200)
        self.fctemp1 = torch.nn.Linear(3200,768)
        self.fctemp2 = torch.nn.Linear(768,num_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()

    def forward(self, x_in, attention_mask, labels):
        gpt_out = self.ptmodel(x_in,attention_mask = attention_mask)[0] #returns tuple
        batch_size = gpt_out.shape[0]
        #print("gpt_out",gpt_out.shape)

        temp = gpt_out.view(batch_size,-1)
        #print("resize = ",temp.shape)

        temp1 = self.fctemp0(temp)
        #print("resize1 = ",temp.shape)

        hidden_vector = self.fctemp1(temp1)
        #print("resize2 = ",hidden_vector.shape)

        prediction_vector = self.fctemp2(hidden_vector)
        #print("resize3 = ",prediction_vector.shape)

        loss = self.loss_func(prediction_vector,labels)
        return loss,prediction_vector,temp  
        

    


def load_two_models(model_path1 = "bert-base-uncased", model_path2 = "bert-base-uncased", nflabels = 2, max_len = 100):
    print('Loading the two model...')

    model1 = LinearClassifier(max_len = max_len, num_classes = nflabels,model_path = model_path1)
    model2 = LinearClassifier(max_len = max_len, num_classes = nflabels,model_path = model_path2)
    
    print("Checking if models are same")
    assert((model1 is model2) == False)

    return model1, model2





def load_model(model_name, nflabels, hidden_size = None):
    print('Loading the model...')
    model = LinearClassifier(hidden_size = hidden_size, num_classes = nflabels,model_path = model_path1)
    return model

def load_tokenizer(model_path = "bert-base-uncased"):
    print('Loading  tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(model_path, progress = False)
    return tokenizer


