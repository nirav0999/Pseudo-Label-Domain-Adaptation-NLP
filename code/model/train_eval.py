import torch
import pandas as pd
import tensorflow as tf
import pickle
import wget
import os
import sys
from torch.utils.data import TensorDataset, random_split
from torch._utils import _accumulate
import torch
import torch.nn as nn
import numpy as np
import time
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
from pathlib import Path
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import classification_report
import argparse
import math
import matplotlib.pyplot as plt
import  sys
import os
from transformers import AdamW


from SeqClass import *
sys.path.insert(0,'../')

from read_write_functions import *


def load_dataset(dataset_path, ids_path):
    """
    Loads the dataset
    """
    ids = loadJsonFile(ids_path)
    complete_dataset = loadJsonFile(dataset_path)
    
    train_df = make_dataset(ids['train'], complete_dataset)
    val_df = make_dataset(ids['val'], complete_dataset)
    test_df = make_dataset(ids['test'], complete_dataset)
    
    return train_df, val_df, test_df

def make_dataset(ids, complete_dataset):

    """
    Curates the dataset in a DataFrame Format
    """
    dataset = []
    
    for commentID  in ids:
        
        label1 = commentID.split('-')[0]; label1 = label1[1:]
        commentinfo = complete_dataset[label1]['data'][commentID]
        
        label2 = commentinfo['classno']
        comment = commentinfo['comment']
        
        assert(commentinfo['nftokens'] >= min_nf_tokens)
        assert(int(label1) == label2)
        
        dataset.append([comment, label2])
    
    print('# Nfsamples = ', len(dataset) - 1)
    
    assert(len(dataset) == len(ids))
    
    df = pd.DataFrame(data = dataset, columns=['sentence', 'label'])
    
    return df

def create_folders(folder_path):
    """
    Hierarchically creates folders
    """
    path = Path(folder_path)
    path.mkdir(parents = True,exist_ok = True)

# Assigns GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def check_gpu():
    # Checking GPU Usage. Get the GPU device name.
    device_name = tf.test.gpu_device_name()
    print(device_name)

    #The device name should look like the following:
    if device_name == '/device:GPU:0':
        print('Found GPU at: {}'.format(device_name))
    else:
        pass
        #raise SystemError('GPU device not found')

    device = None

    if torch.cuda.is_available():       
        device = torch.device("cuda");print('There are %d GPU(s) available.' % torch.cuda.device_count());print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu");print('No GPU available, using the CPU instead.')
    return device

device = check_gpu()

#-----------------PARAMS----------------------------------
# TOKENS taken for a sentence
maximum_length = 100

# Lowercase the sentence for embedding generation
lowercase_sentence = False

# Model Type
model_name1 = "bert-base-uncased1"
model_path1 = "bert-base-uncased"

model_name2 = "bert-base-uncased2"
model_path2 = "bert-base-uncased"


tokenizer_path = "bert-base-uncased"

# Dataset Configurations
nf_train_samples = 7000 
nf_val_samples = 1000
nf_test_samples = 2000
num_classes = 2

# May vary based on size of GPU
min_nf_tokens = 6

# Dumps the model \& results after every epoch
DUMP_MODE = True
loss_type = 'cross_entropy'

# If the labels are one-hot encoded are not
one_hot = False

alpha = 1
    
#------------------------------------------PATHS----------------------------------------------------------
path_str = str(num_classes) + '_' + str(nf_train_samples) + '_' + str(nf_val_samples) + '_' + str(nf_test_samples)
working_folder = "../../"
data_folder = "../../data/labeled/"

#---------Data Folder-----------
source_train_path = data_folder + "source_train.csv"
source_test_path = data_folder + "source_test.csv"
source_val_path = data_folder + "source_val.csv"

# The dataset should necessarily have 'sentence' and 'label' column 
source_train_df, source_val_df, source_test_df = pd.read_csv(source_train_path), pd.read_csv(source_val_path) ,pd.read_csv(source_test_path)

print("Loading Source Data ...........")
print(source_train_df.head(), source_train_df.shape)
print(source_val_df.head(), source_val_df.shape)
print(source_test_df.head(), source_test_df.shape)

target_train_path = data_folder + "target_train.csv"
target_val_path = data_folder + "target_val.csv"
target_path = data_folder + "target_test.csv"

target_train_df, target_val_df, target_test_df = pd.read_csv(target_train_path), pd.read_csv(target_val_path) ,pd.read_csv(target_path)

print("Loading Target Data ...........")
print(target_train_df.head(), target_train_df.shape)
print(target_val_df.head(), target_val_df.shape)
print(target_test_df.head(), target_test_df.shape)

#---------Output folder-----------
store_folder = working_folder + 'results/'

store_folder1 = working_folder + 'results/' + "target_BERT/" + model_name1 + '/' 
store_folder1 += str(min_nf_tokens) + '/'

store_folder2 = working_folder + 'results/' + "source_BERT/" + model_name2 + '/' 
store_folder2 += str(min_nf_tokens) + '/'

models_folder1 = store_folder1  + 'models/' + "target_BERT/" + '/' + loss_type + '/'
model_weights_folder1 = store_folder1 + 'model_weights/' + "target_BERT/" + '/' + loss_type + '/'

models_folder2 = store_folder2 + 'models/' + "source_BERT" + '/' + loss_type + '/'
model_weights_folder2 = store_folder2 + 'model_weights/' + "source_BERT" + '/' + loss_type + '/'

training_stats_folder1 = store_folder + 'per_epoch_stats/' + model_name1 + '/' +  loss_type + '/' 
training_stats_path1 = training_stats_folder1 + path_str + '.json'

training_stats_folder2 = store_folder + 'per_epoch_stats/' + model_name2 + '/' +  loss_type + '/' 
training_stats_path2 = training_stats_folder2 + path_str + '.json'


best_results_folder1 = store_folder1 + 'results/best/' + "target_BERT" + '/' +  loss_type + '/'+ path_str + '/'
best_results_folder2 = store_folder2 + 'results/best/' + "source_BERT" + '/' +  loss_type + '/'+ path_str + '/'

best_predictions_folder1 = store_folder1 + 'predictions/best/' + model_name1 + '/' + loss_type + '/'
best_predictions_folder1 +=  path_str + '/'

best_predictions_folder2 = store_folder2 + 'predictions/best/' + model_name2 + '/' + loss_type + '/'
best_predictions_folder1 +=  path_str + '/'

final_model_path1 = models_folder1 + 'final/' + path_str + '.pkl'
best_model_path1 = models_folder1 + 'best/' + path_str + '.pkl'

final_model_path2 = models_folder2 + 'final/' + path_str + '.pkl'
best_model_path2 = models_folder2 + 'best/' + path_str + '.pkl'

final_model_weights_path1 = model_weights_folder1 + 'final/' + path_str + '.pkl'
best_model_weights_path1 = model_weights_folder1 + 'best/' + path_str + '.pkl'

final_model_weights_path2 = model_weights_folder2 + 'final/' + path_str + '.pkl'
best_model_weights_path2 = model_weights_folder2 + 'best/' + path_str + '.pkl'

#----------------------------------------------------
print("Model1 Name = ",model_name1)
print("Model2 Name = ",model_name2)
print("Number of classes = ",num_classes)
print("Number of train samples per class =",nf_train_samples)
print("Number of val samples per class =",nf_val_samples)
print("Number of test samples per class =",nf_test_samples)
#------------------------------------------------------

if DUMP_MODE == True:
    create_folders(best_results_folder1)
    create_folders(best_results_folder2)

    create_folders(best_predictions_folder1)
    create_folders(best_predictions_folder2)

    create_folders(models_folder1 + 'final/')
    create_folders(models_folder1 + 'best/')
    create_folders(model_weights_folder1 + 'final/')
    create_folders(model_weights_folder1 + 'best/')

    create_folders(models_folder2 + 'final/')
    create_folders(models_folder2 + 'best/')
    create_folders(model_weights_folder2 + 'final/')
    create_folders(model_weights_folder2 + 'best/')
#-----------------------------------------------------


def get_dist(labels, nfclass, verbose = False):
	
    per_class_count = [0 for i in  range(nfclass)]
	
    for label in labels : 
        per_class_count[label] += 1
	
    if verbose == True:
        print('--- Class Distribution ---', per_class_count)

    return per_class_count

def get_one_hot_labels(labels, num_classes = -1, verbose = False):
	one_hot_labels = torch.nn.functional.one_hot(labels, num_classes)
	if verbose == True : print('The shape of One Hot Labels Array = ',one_hot_labels.shape)
	return one_hot_labels


def get_ind_dataset(df):
    
    sentences = df.sentence.values
    labels = list(df.label.values)
    per_class_count = get_dist(labels, num_classes)
    
    labels = np.array(labels)
    labels = torch.from_numpy(labels)
    
    if one_hot == True:
        one_hot_labels = get_one_hot_labels(labels, num_classes)
        labels = one_hot_labels
        
    return sentences, labels

source_train_sentences, source_train_labels = get_ind_dataset(source_train_df)
source_val_sentences, source_val_labels = get_ind_dataset(source_val_df)
source_test_sentences, source_test_labels = get_ind_dataset(source_test_df)

target_train_sentences, target_train_labels = get_ind_dataset(target_train_df)
target_val_sentences, target_val_labels = get_ind_dataset(target_val_df)
target_test_sentences, target_test_labels = get_ind_dataset(target_test_df)

print('Loading Tokenizer .....')
tokenizer = load_tokenizer(tokenizer_path)

print('Loading Models .....')
tModel, sModel = load_two_models(model_path1 = model_path1, model_path2 = model_path2, nflabels = num_classes, max_len = 100)

print(type(tModel))
print(tModel)

print(type(sModel))
print(sModel)

def add_special_token(model, tokenizer,special_token_key = 'pad_token',special_token = '[PAD]'):
    tokenizer.add_special_tokens({special_token_key: special_token})
    model.ptmodel.resize_token_embeddings(len(tokenizer))
    return model,tokenizer

print('If model name is gpt2 or gpt then, special Padding tokens need to be added and the model needs to be made aware of that')
if model_name1 == 'gpt2' or model_name1 == 'openai-gpt':
    tModel, tokenizer = add_special_token(tModel, tokenizer, special_token_key = 'pad_token',special_token = '[PAD]')

if model_name2 == 'gpt2' or model_name2 == 'openai-gpt':
    sModel, tokenizer = add_special_token(sModel, tokenizer, special_token_key = 'pad_token',special_token = '[PAD]')

    
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __add__(self, other):
        return ConcatDataset([self, other])

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    def __len__(self):
        return len(self.indices)

    
def get_additional_info(sentences, labels, lowercase = False):
    input_ids = []
    attention_masks = []
    labels = np.array(labels)
    labels = torch.from_numpy(labels)
    
    for sent in sentences:
        
        sent = str(sent)
        
        if lowercase == True : 
            sent = sent.lower()

        encoded_dict = tokenizer.encode_plus(sent,truncation = True, add_special_tokens = True, 
                                             max_length = maximum_length, pad_to_max_length = True, 
                                             return_attention_mask = True,return_tensors = 'pt')

        input_ids.append(encoded_dict['input_ids'])
        
        attention_masks.append(encoded_dict['attention_mask'])
        
        #max_len = max(max_len, len(encoded_dict['input_ids']))
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)
    labels = torch.tensor(labels)
    
    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])
    print('Attention mask:',attention_masks[0])
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    return dataset

print('Tokenize all of the sentences and map the tokens to their word IDs .....')

source_train_dataset = get_additional_info(source_train_sentences, source_train_labels, lowercase = True)
source_test_dataset = get_additional_info(source_test_sentences, source_test_labels, lowercase = True)
source_val_dataset = get_additional_info(source_val_sentences, source_val_labels, lowercase = True)

target_train_dataset = get_additional_info(target_train_sentences, target_train_labels, lowercase = True)
target_test_dataset = get_additional_info(target_test_sentences, target_test_labels, lowercase = True)
target_val_dataset = get_additional_info(target_val_sentences, target_val_labels, lowercase = True)

def per_class_distribution(dataset, nfclasses = 30,one_hot = False):
    per_class_count = [0 for i in range(nfclasses)]
    Y = [];X = []
    c = 0 
    for sample in dataset:
        y = sample[2].numpy()
        if one_hot == True:
            y = int(np.argmax(np.array(y).reshape(-1,1), axis = 0))
        Y.append(int(y));X.append(1)
        per_class_count[y] += 1
        c += 1
    
    print('------Per class count------')
    return per_class_count,X,Y

print('{:>5,} Source Train samples'.format(len(source_train_dataset)))
print('{:>5,} Source Test samples'.format(len(source_test_dataset)))
print('{:>5,} Source Validation samples'.format(len(source_val_dataset)))

print('{:>5,} Target Train samples'.format(len(target_train_dataset)))
print('{:>5,} Target Test samples'.format(len(target_test_dataset)))
print('{:>5,} Target Validation samples'.format(len(target_val_dataset)))


# The DataLoader needs to know our batch size for training, so we specify it here. 
# For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
batch_size = 16

print('Creating the train dataloader with a batch size ',batch_size,'.....')
# Create the DataLoaders for our training and validation sets. 
source_train_dataloader = DataLoader(source_train_dataset, sampler = RandomSampler(source_train_dataset), batch_size = batch_size, shuffle = False)

# Sequential Sampler is used instead of Random Sampler
# The batch size of test dataloader is nf_test_samples * num_classes to process the whole thing as 1 batch
print('Creating the test dataloader with a batch size',nf_test_samples * num_classes ,'.....')
source_test_dataloader = DataLoader(source_test_dataset, sampler = SequentialSampler(source_test_dataset), batch_size = 1 , shuffle = False)

print('Creating the val dataloader with a batch size.....')
source_val_dataloader = DataLoader(source_val_dataset, sampler = SequentialSampler(source_val_dataset), batch_size = 1 , shuffle = False)

# Create the DataLoaders for our training and validation sets. 
target_train_dataloader = DataLoader(target_train_dataset, sampler = SequentialSampler(target_train_dataset), batch_size = 1, shuffle = False)

# Sequential Sampler is used instead of Random Sampler
# The batch size of test dataloader is nf_test_samples * num_classes to process the whole thing as 1 batch
print('Creating the test dataloader with a batch size',nf_test_samples * num_classes ,'.....')
target_val_dataloader = DataLoader(target_val_dataset, sampler = SequentialSampler (target_val_dataset), batch_size = 1 , shuffle = False)

print('Creating the val dataloader with a batch size.....')
target_test_dataloader = DataLoader(target_test_dataset, sampler = SequentialSampler(target_test_dataset), batch_size = 1 , shuffle = False)


print("Summary of the model's parameters as a list of tuples.")
params1 = list(tModel.named_parameters())
params2 = list(sModel.named_parameters())

print('The tModel has {:} different named parameters.\n'.format(len(params1)))
print('==== Embedding Layer ====\n')
for p in params1[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params1[5:-4]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params1[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


print('The sModel has {:} different named parameters.\n'.format(len(params2)))
print('==== Embedding Layer ====\n')
for p in params2[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== First Transformer ====\n')
for p in params2[5:-4]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
print('\n==== Output Layer ====\n')
for p in params2[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer1 = AdamW(tModel.parameters(),
				lr = 5e-5, # default is 5e-5,
				eps = 1e-8 # default is 1e-8.
                )

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer2 = AdamW(sModel.parameters(),
				lr = 5e-5, # default is 5e-5,
				eps = 1e-8 # default is 1e-8.
                )

# Number of training epochs. The BERT authors recommend between 2 and 4. I ran for maximum 7
epochs = 2

# Total number of training steps is [number of batches] x [number of epochs]. (Note that this is not the same as the number of training samples).
total_steps = len(source_train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps = 0, num_training_steps = total_steps)
scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps = 0, num_training_steps = total_steps)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    """
    Calculates the accuracy
    """
    pred_flat = np.argmax(preds, axis = 1).flatten()
    labels_flat = labels.flatten()
    results = classification_report(labels_flat, pred_flat, digits = 5,output_dict = False)
    results_json = classification_report(labels_flat, pred_flat, digits = 5,output_dict = True)
    accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)
    return accuracy, results, results_json, pred_flat 

def format_time(elapsed):
	'''
	Takes a time in seconds and returns a string hh:mm:ss
	'''
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))
	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds=elapsed_rounded))


# model = model.cuda()
tModel = tModel.cuda()
sModel = sModel.cuda()
device = check_gpu()

# This training code is based on the `run_glue.py` script here: https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

def seed_everything(seed):
	random.seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True

seed_everything(seed_val) 

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

model_after_2_epochs = None
best_model = None
best_accuracy = 0
best_loss = 9999999999999
criterion = None

# if loss_type == 'multilabel':
#     criterion = torch.nn.MultiLabelSoftMarginLoss()


def organize_embeddings(hidden_states):
    final_embeddings = []
    for batch_embeddings in hidden_states:
        batch_embed_numpy = batch_embeddings
        for embed in batch_embed_numpy: 
                final_embeddings.append(embed)
    final_embeddings = np.array(final_embeddings).reshape(-1,1)
    return final_embeddings
    
def evaluate(model, dataloader):
    """
    Evaluating the model
    """

    
    print('Evaluating the model ......')
    
    # Start time
    t0 = time.time()
    
    # Put the model in evaluation mode-- the dropout layers behave differently during evaluation.
    model.eval()
    
    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    # Initializing all storage Units
    all_logits = [];all_label_ids = [];all_hidden_states = []
    
    # Batch
    batch_no = 0
    
    for batch in dataloader:
        
        # Loading into Device (GPU)
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2]
        
        with torch.no_grad():
            if loss_type == 'multilabel': 
                b_labels = np.argmax(b_labels, axis = 1)
                
            b_labels = b_labels.to(device)
            # Loading into model ang getting the results
            loss, logits, hidden = model(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
            
        # Evaluation Loss 
        total_eval_loss += loss.item()
        
        # Logits 
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        all_hidden_states.append(hidden[-1].to('cpu').numpy())
        
        if batch_no == 0 :
            all_label_ids, all_logits = label_ids, logits
        else:
            all_label_ids = np.concatenate((all_label_ids,label_ids), axis = 0)
            all_logits = np.concatenate((all_logits, logits), axis = 0)
            #all_hidden_states = np.concatenate((all_hidden_states,logits), axis = 0)
        
        # if batch_no % 10000 == 0 :
        #     print(batch_no, all_label_ids.shape, all_logits.shape, len(all_hidden_states),all_hidden_states[0].shape)
            #print(batch_no, all_label_ids.shape, all_logits.shape)
        
        batch_no += 1 
    

    final_embeddings = organize_embeddings(all_hidden_states)
    
    # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches.
    eval_accuracy, results, results_json, predictions = flat_accuracy(all_logits, all_label_ids)
    accuracy = results_json['accuracy']
    print(results)
    
    # Calculate the average loss over all of the batches.
    loss = total_eval_loss / len(dataloader)
    
    # Measure how long the validation run took.
    end_time = time.time() 
    test_time = format_time(end_time - t0)
    
    print("\t Accuracy: {0:.5f}".format(accuracy))
    print("\t Loss: {0:.5f}".format(loss))
    print("\t Timing took: {:}".format(test_time))
    print("\t Confirming Accuracy:", eval_accuracy)
    print("\t Number of Samples :")
    print(label_ids.shape, logits.shape, final_embeddings.shape)
    
    return results_json, predictions, final_embeddings, all_label_ids, all_logits, loss, test_time 
    

# For each epoch...
for epoch_i in range(0, epochs):
    # ========================================
    #               Training
    # ========================================
    # Perform one full pass over the training set
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss1 = 0
    total_train_loss2 = 0
    total_weight_diff_loss = 0
    total_train_loss_combined = 0

    # Put the model into training mode. Don't be mislead--the call to `train` just changes the *mode*, 
    # it doesn't *perform* the training. `dropout` and `batchnorm` layers behave differently during training vs. test 
    #(source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    # model.train()
    tModel.train()
    sModel.train()

    for step, batch in enumerate(source_train_dataloader):
        # Progress update every 40 batches.
        if step % 100 == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(source_train_dataloader), elapsed))
        
        # Unpack this training batch from our dataloader. 
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method. `batch` contains three pytorch tensors: [0]: input ids, [1]: attention masks, [2]: labels 

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2]

        # print(b_input_ids.shape,b_labels.shape)
        # Always clear any previously calculated gradients before performing a backward pass. 
        # PyTorch doesn't do this automatically because accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)

        tModel.zero_grad()
        sModel.zero_grad()
        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments arge given and what flags are set. 
        # For our useage here, it return the loss (because we provided labels) 
        # and the "logits"--the model outputs prior to activation.
        
        if loss_type == 'multilabel' : b_labels = np.argmax(b_labels, axis = 1)
        b_labels = b_labels.to(device)
        
        loss1, logits1, hidden1 = tModel(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
        loss2, logits2, hidden2 = sModel(b_input_ids, attention_mask = b_input_mask, labels = b_labels)
        

        # For multilabel loss
        # if loss_type == 'multilabel':
        #     b_labels = b_labels.cpu()
        #     b_labels = get_one_hot_labels(b_labels, num_classes)
        #     b_labels = b_labels.to(device)
        #     loss = criterion(logits, b_labels)
        
        # Accumulate the training loss over all of the batches so that we can calculate the average loss at the end. 
        # `loss` is a Tensor containing a single value; the `.item()` function just returns the Python value from the tensor.
        # total_train_loss += loss.item()
        total_train_loss1 += loss1.item()
        total_train_loss2 += loss2.item()

        # break
        
        # params1_x = tModel.parameters()

        last_output1_weights = None
        all_params = []
        for param_no, param in enumerate(tModel.parameters()):
            #print(param_no, param.shape)
            all_params.append(param)  

        last_output1_weights = all_params[-2]

        last_output2_weights = None
        all_params = []
        for param_no, param in enumerate(sModel.parameters()):
            #print(param_no, param.shape)
            all_params.append(param)  

        last_output2_weights = all_params[-2]

        last_output2_weights = torch.transpose(last_output2_weights, 0, 1)
        weighted_loss = torch.matmul(last_output1_weights, last_output2_weights)
        weighted_loss = torch.abs(weighted_loss)
        column_wise_sum = torch.sum(weighted_loss, 0)
        weight_diff = torch.mean(column_wise_sum)

        # hidden1 = torch.transpose(hidden1, 0, 1)
        hidden_loss = hidden1 - hidden2
        hidden_loss = torch.abs(hidden_loss)
        hidden_loss_column_wise_sum = torch.sum(hidden_loss, 0)
        # hidden_diff = torch.mean(hidden_loss_column_wise_sum)


        total_weight_diff_loss += weight_diff.item()

        # print(hidden_diff)
        # print(hidden_diff.shape)
        
        # print(weight_diff)
        # print(weight_diff.shape)

        # print("---------------------")

        total_train_loss = loss1 + loss2 + alpha * weight_diff
        
        total_train_loss.backward()

        
        # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(tModel.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(sModel.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient. 
        # The optimizer dictates the "update rule"--how the parameters are modified based on their gradients, 
        # the learning rate, etc.

        optimizer1.step()
        scheduler1.step() # Update the learning rate.

        optimizer2.step()
        scheduler2.step() # Update the learning rate.

        


    # Calculate the average loss over all of the batches.
    train_loss = total_train_loss / len(source_train_dataloader)
    train_loss1 = total_train_loss1 / len(source_train_dataloader)
    train_loss2 = total_train_loss2 / len(source_train_dataloader)

    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    
    print("  Average training loss: {0:.2f}".format(train_loss))
    print("  Training epoch took: {:}".format(training_time))

    # ========================================
    #               Test
    # ========================================
    # After the completion of each training epoch, measure our performance our Test set.
    
    print('Evaluating source train dataset ....')
    t_on_s_train_results_json, _, t_on_s_train_final_embeddings, t_on_s_train_label_ids, t_on_s_train_logits, _, _ = evaluate(tModel, source_train_dataloader)
    s_on_s_train_results_json, _, s_on_s_final_embeddings, s_on_s_label_ids, s_on_s_logits, _, _ = evaluate(sModel, source_train_dataloader)

    print('Evaluating source validation dataset ....')
    t_on_s_val_results_json, _, t_on_s_final_embeddings, t_on_s_label_ids, t_on_s_logits, _, _ = evaluate(tModel, source_val_dataloader)
    s_on_s_val_results_json, _, s_on_s_final_embeddings, s_on_s_label_ids, s_on_s_logits, _, _ =  evaluate(sModel, source_val_dataloader)
    
    print('Evaluating source test dataset ....')
    t_on_s_test_results_json, _, t_on_s_test_final_embeddings, t_on_s_test_label_ids, t_on_s_test_logits, _, _ = evaluate(tModel, source_test_dataloader)
    s_on_s_test_results_json, _, s_on_s_test_final_embeddings, s_on_s_test_label_ids2, s_on_s_test_logits, _, _ = evaluate(sModel, source_test_dataloader)
    
    print('Evaluating target train dataset ....')
    t_on_t_train_results_json, _, t_on_t_train_final_embeddings, t_on_t_train_label_ids, t_on_t_train_logits, _, _ = evaluate(tModel, target_train_dataloader)
    s_on_t_train_results_json, _, s_on_t_final_embeddings, s_on_t_label_ids, s_on_t_logits, _ , _ = evaluate(sModel, target_train_dataloader)
    
    print('Evaluating target validation dataset ....')
    t_on_t_val_results_json, _, t_on_t_val_final_embeddings, t_on_t_val_label_ids, _, _ , _ = evaluate(tModel, target_val_dataloader)
    s_on_t_val_results_json, _, s_on_t_val_final_embeddings, s_on_t_val_label_ids, s_on_t_val_logits, _, _ =  evaluate(sModel, target_val_dataloader)
    
    print('Evaluating target test dataset ....')
    t_on_t_test_results_json, _, t_on_t_test_final_embeddings, t_on_t_test_label_ids, t_on_t_test_logits, _, _ = evaluate(tModel, target_test_dataloader)
    s_on_t_test_results_json, _, s_on_t_test_final_embeddings, s_on_t_test_label_ids, s_on_t_test_logits, _, _ = evaluate(sModel, target_test_dataloader)
    
    t_on_s_train_accuracy = t_on_s_train_results_json['accuracy']
    t_on_s_test_accuracy = t_on_s_test_results_json['accuracy']
    t_on_s__val_accuracy = t_on_s_train_results_json['accuracy']

    s_on_s_train_accuracy = s_on_s_train_results_json['accuracy']
    s_on_s_test_accuracy = s_on_s_test_results_json['accuracy']
    s_on_s_val_accuracy = s_on_s_val_results_json['accuracy']

    t_on_t_train_accuracy = t_on_t_train_results_json['accuracy']
    t_on_t_test_accuracy = t_on_t_test_results_json['accuracy']
    t_on_t_accuracy = t_on_t_val_results_json['accuracy']

    s_on_t_train_accuracy = s_on_t_train_results_json['accuracy']
    s_on_t_test_accuracy = s_on_t_test_results_json['accuracy']
    s_on_t_val_accuracy = s_on_t_val_results_json['accuracy']

        
    if DUMP_MODE == True:
        dumpPickleFile([t_on_t_test_label_ids, t_on_t_test_final_embeddings,
                        t_on_t_test_logits],
                        best_predictions_folder1 +'epoch_'+ str(epoch_i)+'_t_on_t_predictions_test.pkl')

        dumpPickleFile([t_on_t_val_label_ids1, t_on_t_val_final_embeddings1,
                        t_on_t_val_logits],
                        best_predictions_folder1 +'epoch_'+ str(epoch_i)+'_t_on_t_predictions_val.pkl')

        dumpPickleFile([t_on_t_train_label_ids, t_on_t_train_final_embeddings,
                        t_on_t_train_logits],
                        best_predictions_folder1+'epoch_'+str(epoch_i)+'_t_on_t_predictions_train.pkl')
        
        dumpPickleFile([s_on_t_test_label_ids2, s_on_t_test_final_embeddings2,
                        s_on_t_test_logits2],
                        best_predictions_folder2 +'epoch_'+ str(epoch_i)+'_s_on_t_predictions_test.pkl')

        dumpPickleFile([s_on_t_val_label_ids2, s_on_t_val_final_embeddings2,
                        s_on_t_val_logits2],
                        best_predictions_folder2 +'epoch_'+ str(epoch_i)+'_s_on_t_predictions_val.pkl')

        dumpPickleFile([s_on_t_train_label_ids,s_on_t_train_final_embeddings,
                        s_on_t_train_logits],
                        best_predictions_folder2+'epoch_'+str(epoch_i)+'_s_on_t_predictions_train.pkl')
        
        
        torch.save(tModel.state_dict(), best_model_weights_path1)
        torch.save(tModel, best_model_path1)

        torch.save(sModel.state_dict(), best_model_weights_path2)
        torch.save(sModel, best_model_path1)

    print("------------- Accuracies -------------")

    print("=== Target BERT model on Target dataset ===")
    print("Train accuracy: ", t_on_t_train_accuracy)
    print("Test accuracy: ", t_on_t_test_accuracy)
    print("Val accuracy: ", t_on_t_accuracy)


    print("=== Source BERT model on Target dataset ===")
    print("Train accuracy: ", s_on_t_train_accuracy)
    print("Test accuracy: ", s_on_t_test_accuracy)
    print("Val accuracy: ", s_on_t_val_accuracy)


    training_stats = {
            'epoch': epoch_i + 1,
            'Total Training loss': train_loss.item(),
            'CE Target Training loss': t_on_s_train_loss,
            'CE Source Training loss': s_on_s_train_loss,
            'Weight_diff_loss': total_weight_diff_loss,

            'Source on Source test accuracy': s_on_s_test_accuracy,
            'Target on Source test accuracy': t_on_s_test_accuracy,
            'Target on Target test accuracy': t_on_t_test_accuracy,
            'Source on Target test accuracy': s_on_t_test_accuracy,

            'Source on Source val accuracy': s_on_s_val_accuracy,
            'Target on Source val accuracy': t_on_s_val_accuracy,
            'Target on Target val accuracy': t_on_t_val_accuracy,
            'Source on Target val accuracy': s_on_t_val_accuracy,

            'Source on Source train accuracy': s_on_s_train_accuracy,
            'Target on Source train accuracy': t_on_s_train_accuracy,
            'Target on Target train accuracy': t_on_t_train_accuracy,
            'Source on Target train accuracy': s_on_t_train_accuracy
        }

    print(training_stats)

    print('Dumping training folders .....')
    
    # if DUMP_MODE == True:
    #     dumpJsonFile(training_stats, training_stats_path)
    dumpJsonFile(training_stats, working_folder + "training_results.json")


print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
