import torch
import random
from util import cudaify, clear_cuda
from transformers import BertTokenizer, BertModel
from networks import DropoutClassifier
from readdata import read_from_training_data


class BertForWordSegmentation(torch.nn.Module):
    def __init__(self, window_size):
        super(BertForWordSegmentation, self).__init__()
        self.window_size = window_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = cudaify(BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True))
        self.classifier = cudaify(DropoutClassifier(768 * self.window_size, 2))
        
    def forward(self, input_tokens, labels = None):
        bert_tokens = []
        bert_tokens.append("[CLS]")
        bert_tokens += input_tokens
        bert_tokens.append("[SEP]")
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = cudaify(torch.tensor([indexed_tokens]))
        outputs = self.model(tokens_tensor)
        pooled_output = outputs[2]
        processed_list = []
        assert(len(indexed_tokens) == pooled_output[0].shape[1])
        for i in range(1, len(indexed_tokens) - 2):
            means = []
            for w in range(self.window_size):
                mean = (pooled_output[0][0][i + w] + pooled_output[12][0][i + w]) / 2
                means.append(mean)
                
            #mean_1 = (pooled_output[0][0][i] + pooled_output[12][0][i]) / 2
            #mean_2 = (pooled_output[0][0][i + 1] + pooled_output[12][0][i + 1]) / 2
            processed_list.append(torch.unsqueeze(torch.cat(means, 0), 0))
        processed_tensor = cudaify(torch.cat(processed_list, 0))
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = cudaify(torch.LongTensor(labels[:(len(labels) - 1)])) # final label is always 1, hence ignored
            loss = loss_fct(result, y)
        return result, loss


def data_loader(x_list, y_list):
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)

    for i in range(len(x_tuple)):
        yield x_tuple[i], y_tuple[i]

def train(x_train, y_train, x_dev, y_dev, model, num_epochs, learning_rate, 
          do_save, save_path, eliminate_one):
    best_model = model
    best_acc = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        train_loader = data_loader(x_train, y_train)
        test_loader = data_loader(x_dev, y_dev)
        batch_cnt = 0
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            z, loss = model(x, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data.item()
            batch_cnt += 1
        print ('Epoch: [%d/%d], Average Loss: %.4f' % (epoch+1, num_epochs, total_loss / batch_cnt))
            
        num_characters = 0
        correct_predictions = 0
        for x, y in test_loader:
            z, _ = model(x)
            for (i, output) in enumerate(z):
                if y[i] == output.argmax():
                    correct_predictions += 1
                num_characters += 1
        print('Test Accuracy: %.4f' % (correct_predictions * 1.0 / num_characters))
        if correct_predictions * 1.0 / num_characters > best_acc:
            best_acc = correct_predictions * 1.0 / num_characters
            best_model = model
        clear_cuda()
    if do_save:
        torch.save(best_model, save_path)

def character_stream(filename, num_lines):
    with open(filename) as inhandle:
        for i in range(num_lines):
            line = inhandle.readline()
            for char in line:
                yield char
        
def train_2(train_file, dev_file, num_sentences, window_size=2, 
            num_epochs = 15, learning_rate = 0.005, do_save = True, 
            save_path = 'FineTuneModel.bin', eliminate_one = True):
    
    
    x_train, y_train = read_from_training_data(character_stream(train_file, num_sentences))
    x_dev, y_dev = read_from_training_data(character_stream(dev_file, num_sentences))
    model = BertForWordSegmentation(window_size)
    train(x_train, y_train, x_dev, y_dev, model, num_epochs, learning_rate, 
          do_save, save_path, eliminate_one)
    clear_cuda()
