import torch
import random
from util import cudaify, clear_cuda
from transformers import BertTokenizer, BertModel
from networks import DropoutClassifier
from readdata import read_train_data
from segmenter import segment_file, XE, BMES
from embed import GapEmbedder
from embed import SimpleEmbedder, GapAverageEmbedder, WideEmbedder
import copy

class BertForWordSegmentation(torch.nn.Module):
    def __init__(self, embedder, encoding, bert_model = 'bert-base-chinese'):
        super(BertForWordSegmentation, self).__init__()
        self.embedder = embedder
        self.encoding = encoding
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case = False)
        self.model = cudaify(BertModel.from_pretrained(bert_model, output_hidden_states=True))
        self.classifier = cudaify(DropoutClassifier(self.embedder.embedding_width(), self.encoding.domain_size()))
        
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
            processed_list.append(self.embedder(pooled_output, i))
        processed_tensor = cudaify(torch.cat(processed_list, 0))
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = cudaify(torch.LongTensor(labels[:(len(labels) - 1)])) # final label is always 1, hence ignored
            loss = loss_fct(result, y)
        return result, loss
    
    def segmenter(self):
        def segment(sent):
            z, _ = self([tok for tok in sent])
            flags = []
            for i in range(len(z)):
                flags.append(z[i].argmax().item())                
            flags.append(self.encoding.terminator())
            return flags
        return segment
                
    
def data_loader(x_list, y_list):
    z = list(zip(x_list, y_list))
    random.shuffle(z)
    x_tuple, y_tuple = zip(*z)

    for i in range(len(x_tuple)):
        yield x_tuple[i], y_tuple[i]


def train(x_train, y_train, x_dev, y_dev, model, num_epochs, learning_rate, 
          save_path):
    best_model = model
    best_acc = 0.0
    
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print("Starting first epoch...")
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
            print("Updating best model.")
            best_acc = correct_predictions * 1.0 / num_characters
            best_model = copy.deepcopy(model)
        clear_cuda()
    if save_path is not None:
        torch.save(best_model, save_path)
    return best_model


def line_stream(filename, num_lines):
    with open(filename) as inhandle:
        for i in range(num_lines):
            line = inhandle.readline()
            yield line.strip()
        
def main(train_file, dev_file, test_file):
   
    def run_experiment(num_sentences, 
                       text_output_file,
                       embedder,
                       encoding,
                       model_file = 'FineTuneModel.bin', 
                       num_epochs = 15, 
                       learning_rate = 0.005):
            
        model = BertForWordSegmentation(embedder, encoding)
        x_train, y_train = read_train_data(line_stream(train_file, num_sentences), encoding)
        x_dev, y_dev = read_train_data(line_stream(dev_file, num_sentences), encoding)
        net = train(x_train, y_train, x_dev, y_dev, model, num_epochs, learning_rate, 
                    model_file)
        segment_file(net, test_file, text_output_file)
        clear_cuda()

    BERT_EMBEDDING_WIDTH = 768
    for trial in range(3):
        for i in [500, 1000, 2000, 4000, 8000, 16000]:
            run_experiment(i, 'gap.{}.{}.xe.txt'.format(i, trial), 
                           GapEmbedder(BERT_EMBEDDING_WIDTH),
                           XE(),
                           model_file = 'gap.{}.{}.xe.bin'.format(i, trial),
                           num_epochs = 10)


    """
    run_experiment(18500, 'gap.19k.bmes.txt', 
                   GapEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = 'gap.19k.bmes.bin',
                   num_epochs = 10)                   
    run_experiment(18500, 'gap.19k.bmes.2.txt', 
                   GapEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = 'gap.19k.bmes.2.bin',
                   num_epochs = 10)                   

    run_experiment(2000, 'gap.2k.bmes.txt', 
                   GapEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = None,
                   num_epochs = 10)
    run_experiment(18500, 'gap.19k.xe.txt', 
                   GapEmbedder(BERT_EMBEDDING_WIDTH),
                   XE(),
                   model_file = 'gap.19k.xe.bin',
                   num_epochs = 10)                   
    run_experiment(18500, 'gap.19k.bmes.txt', 
                   GapEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = 'gap.19k.bmes.bin',
                   num_epochs = 10)                   
    run_experiment(18500, 'wide.19k.xe.txt', 
                   WideEmbedder(BERT_EMBEDDING_WIDTH),
                   XE(),
                   model_file = 'wide.19k.xe.bin',
                   num_epochs = 10)                   
    run_experiment(18500, 'wide.19k.bmes.txt', 
                   WideEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = 'wide.19k.bmes.bin',
                   num_epochs = 10)                   
    run_experiment(18500, 'simple.19k.xe.txt', 
                   SimpleEmbedder(BERT_EMBEDDING_WIDTH),
                   XE(),
                   model_file = 'simple.19k.xe.bin',
                   num_epochs = 10)
    run_experiment(18500, 'simple.19k.bmes.txt', 
                   SimpleEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = 'simple.19k.bmes.bin',
                   num_epochs = 10)
    run_experiment(18500, 'gapavg.19k.xe.txt', 
                   GapAverageEmbedder(BERT_EMBEDDING_WIDTH),
                   XE(),
                   model_file = 'gapavg.19k.xe.bin',
                   num_epochs = 10)                   
    run_experiment(18500, 'gapavg.19k.bmes.txt', 
                   GapAverageEmbedder(BERT_EMBEDDING_WIDTH),
                   BMES(),
                   model_file = 'gapavg.19k.bmes.bin',
                   num_epochs = 10)     
    """


    
if __name__ == '__main__':
    net = main('data/bakeoff/training/pku_training.utf8', 
               'pku_dev.utf8',               
               'data/bakeoff/testing/pku_test.utf8')
    
