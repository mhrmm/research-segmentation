# -*- coding: utf-8 -*-
import torch
import pandas as pd

from train import train_net
from networks import DropoutClassifier
from util import cudaify
from readdata import read_from_testing_data
from sklearn.linear_model import LogisticRegression


def tensor_batcher(t, batch_size, balanced=True):
    def shuffle_rows(a):
        return a[torch.randperm(a.size()[0])]        
    neg = t[(t[:, 0] == 0).nonzero().squeeze(1)] # only negative rows
    pos = t[(t[:, 0] == 1).nonzero().squeeze(1)] # only positive rows
    neg = shuffle_rows(neg)
    pos = shuffle_rows(pos)
    if balanced:
        min_row_count = min(neg.shape[0], pos.shape[0])
        epoch_data = torch.cat([neg[:min_row_count], pos[:min_row_count]])    
    else:
        epoch_data = torch.cat([neg, pos])            
    epoch_data = shuffle_rows(epoch_data)
    for i in range(0, len(epoch_data), batch_size):
        yield epoch_data[i:i+batch_size]
        
def test_training(d, k):
    def nth_dim_positive_data(n, d, k):
        data = torch.randn(d, k)
        u = torch.cat([torch.clamp(torch.sign(data[2:3]), min=0), data])
        return u.t()

    train = nth_dim_positive_data(2, d, k)
    dev = nth_dim_positive_data(2, d, 500)
    #test = nth_dim_positive_data(2, d, 500)   
    classifier = DropoutClassifier(d,100,2)
    train_net(classifier, train, dev, tensor_batcher,
              batch_size=96, n_epochs=30, learning_rate=0.001,
              verbose=True)


def create_and_train_net(training_data, test_data):
    training_data = cudaify(training_data)
    test_data = cudaify(test_data)
    print("training size:", training_data.shape)
    print("testing size:", test_data.shape)
    classifier = cudaify(DropoutClassifier(1536, 2, 200))
    return train_net(classifier, training_data, test_data, 
                     lambda x,y: tensor_batcher(x,y,False),
                     batch_size=96, n_epochs=12, learning_rate=0.001,
                     verbose=True)
    
def train_from_csv(train_csv, dev_csv):
    print('loading train')
    train = torch.tensor(pd.read_csv(train_csv).values).float()
    print('train size: {}'.format(train.shape[0]))
    print('loading dev')
    dev = torch.tensor(pd.read_csv(dev_csv).values).float()
    print('dev size: {}'.format(dev.shape[0]))
    net = create_and_train_net(train, dev)
    net.eval()
    return net

def segment_test_file(model, embedder, test_file, 
                      output_filename = 'result.txt'):
    characters = open(test_file).read()   
    characters = [ch for ch in characters if ch != ' ']
    x_list = read_from_testing_data(test_file)    
    output = open(output_filename, "w+")
    sentence_id = 0
    token_id = 0
    character_id = 0
    for j, tokens in enumerate(x_list):
        if j % 100 == 0:
            print('{}/{}'.format(j, len(x_list)))
        x_tensor = embedder(tokens, [-1 for _ in range(len(tokens))])
        x_tensor = x_tensor[:,1:]
        z = model(x_tensor)

        for i, tok in enumerate(tokens):
            # if this is the last token:
            if i == len(tokens) - 1:
                output.write(characters[character_id])
                output.write("  ")
                if characters[character_id + 1] == '\n':
                    output.write("\n")
                    character_id += 1
                token_id = 0
                sentence_id += 1
                character_id += 1
            else:
                output.write(characters[character_id])
            
                if z[i].argmax().item() == 1:
                    output.write("  ")
                token_id += 1
                character_id += 1
    if character_id < len(characters):
        output.write(characters[character_id])
    output.write("  \n")
    output.close()    


def train_and_eval_from_csv(training_csv, dev_csv, embedder, 
                            test_file, output_filename = 'result.txt'):
    model = train_from_csv(training_csv, dev_csv)    
    model.eval()
    characters = open(test_file).read()    
    x_list = read_from_testing_data(test_file)    
    output = open(output_filename, "w+")
    sentence_id = 0
    token_id = 0
    character_id = 0
    for j, tokens in enumerate(x_list):
        if j % 1000 == 0:
            print('{}/{}'.format(j, len(x_list)))
        x_tensor = embedder(tokens, [-1 for _ in range(len(tokens))])
        x_tensor = x_tensor[:,1:]
        z = model(x_tensor)

        for i, tok in enumerate(tokens):
            # if this is the last token:
            if i == len(tokens) - 1:
                output.write(characters[character_id])
                output.write("  ")
                if characters[character_id + 1] == '\n':
                    output.write("\n")
                    character_id += 1
                token_id = 0
                sentence_id += 1
                character_id += 1
            else:
                output.write(characters[character_id])
            
                if z[i].argmax().item() == 1:
                    output.write("  ")
                token_id += 1
                character_id += 1
    output.write(characters[character_id])
    output.write("  \n")
    output.close()


def train_lr(filename):

    data = torch.tensor(pd.read_csv(filename).values).float()

    x_train = data[:,1:].numpy()
    y_train = torch.clamp(data[:,0], min=0).long().numpy()
    num_train = int(x_train.shape[0] * 0.8)
    x_test = x_train[num_train:]
    y_test = y_train[num_train:]
    x_train = x_train[:num_train]
    y_train = y_train[:num_train]
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(sum(y_train))

    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
  
    #x_test = x_train
    #y_test = y_train
    predictions = logisticRegr.predict(x_train)    
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == y_train[i]:
            count += 1
    print('Train accuracy: {}'.format(count / len(y_train)))
    predictions = logisticRegr.predict(x_test)    
    count = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            count += 1
    print('Test accuracy: {}'.format(count / len(y_test)))
    