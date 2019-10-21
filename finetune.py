import torch
import random
from util import cudaify, clear_cuda
from transformers import BertTokenizer, BertModel
from networks import DropoutClassifier
from readdata import read_from_training_data, read_from_testing_data, is_english

class Embedder:

    def __init__(self, base_embedding_width):
        self.base_embedding_width = base_embedding_width

    def embedding_width(self):
        raise NotImplementedError("Cannot call a generic Embedder.")
    
    def __call__(self, layers, i):
        raise NotImplementedError("Cannot call a generic Embedder.")

class SimpleEmbedder(Embedder):
    def __init__(self, base_embedding_width):
        super().__init__(base_embedding_width)

    def embedding_width(self):
        return self.base_embedding_width

    def __call__(self, layers, i):
        means = []
        for w in range(1):
            mean = (layers[0][0][i + w] + layers[12][0][i + w]) / 2
            means.append(mean)
        return torch.unsqueeze(torch.cat(means, 0), 0)

class GapEmbedder(Embedder):
    def __init__(self, base_embedding_width):
        super().__init__(base_embedding_width)

    def embedding_width(self):
        return 2 * self.base_embedding_width

    def __call__(self, layers, i):
        means = []
        for w in range(2):
            mean = (layers[0][0][i + w] + layers[12][0][i + w]) / 2
            means.append(mean)
        result = torch.unsqueeze(torch.cat(means, 0), 0)
        return result

class GapAverageEmbedder(Embedder):
    def __init__(self, base_embedding_width):
        super().__init__(base_embedding_width)

    def embedding_width(self):
        return self.base_embedding_width

    def __call__(self, layers, i):
        mean = (layers[0][0][i] + layers[12][0][i] + layers[0][0][i+1] + layers[12][0][i+1]) / 4
        result = torch.unsqueeze(mean, 0)
        return result

class WideEmbedder(Embedder):
    def __init__(self, base_embedding_width):
        super().__init__(base_embedding_width)
        
    def embedding_width(self):
        return 4 * self.base_embedding_width

    def __call__(self, layers, i):
        means = []
        for w in range(-1, 3):
            mean = (layers[0][0][i + w] + layers[12][0][i + w]) / 2
            means.append(mean)
        return torch.unsqueeze(torch.cat(means, 0), 0)

class BertForWordSegmentation(torch.nn.Module):
    def __init__(self):
        super(BertForWordSegmentation, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased', do_lower_case = False, output_hidden_states=True).to('cuda')
        self.classifier = DropoutClassifier(768 * 2, 2).to('cuda')
        
    def forward(self, input_tokens, labels = None):
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(input_tokens)
        tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
        outputs = self.model(tokens_tensor)
        pooled_output = outputs[2]
        processed_list = []
        assert(len(input_tokens) == pooled_output[0].shape[1])
        for i in range(len(indexed_tokens) - 1):
            mean_1 = (pooled_output[0][0][i] + pooled_output[12][0][i]) / 2
            mean_2 = (pooled_output[0][0][i + 1] + pooled_output[12][0][i + 1]) / 2
            processed_list.append(torch.unsqueeze(torch.cat((mean_1, mean_2), 0), 0))
        processed_tensor = torch.cat(processed_list, 0).to('cuda')
        result = self.classifier(processed_tensor)
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            y = torch.LongTensor(labels[:(len(labels) - 1)]).to('cuda')
            loss = loss_fct(result, y)
        return result, loss


class BertForWordSegmentationNew(torch.nn.Module):
    def __init__(self, embedder):
        super(BertForWordSegmentation, self).__init__()
        self.embedder = embedder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
        self.model = cudaify(BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True))
        self.classifier = cudaify(DropoutClassifier(self.embedder.embedding_width(), 2))
        
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
    return best_model



def segment_test_file(model, test_file, 
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
        if len(tokens) > 1:
            z, loss = model(tokens)

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
    
    
def character_stream(filename, num_lines):
    with open(filename) as inhandle:
        for i in range(num_lines):
            line = inhandle.readline()
            for char in line:
                yield char
        
def main(train_file, dev_file, num_sentences, window_size=2, 
         num_epochs = 15, learning_rate = 0.005, do_save = True, 
         save_path = 'FineTuneModel.bin', eliminate_one = True):
    
    BERT_EMBEDDING_WIDTH = 768
    x_train, y_train = read_from_training_data(character_stream(train_file, num_sentences))
    x_dev, y_dev = read_from_training_data(character_stream(dev_file, num_sentences))
    #model = BertForWordSegmentation(GapAverageEmbedder(BERT_EMBEDDING_WIDTH))
    model = BertForWordSegmentation()
    net = train(x_train, y_train, x_dev, y_dev, model, num_epochs, learning_rate, 
                do_save, save_path, eliminate_one)
    segment_test_file(net, dev_file, 'result.txt')
    clear_cuda()

def prepare_script(model_path, test_file, output_filename = 'FineTune_result.txt'):
    model = torch.load(model_path)
    
    test_chars = open(test_file).read()
    characters = []
    hash_string = ""
    for c in test_chars:
        if is_english(c):
            hash_string += c
        else:
            if hash_string != "":
                characters.append(hash_string)
                hash_string = ""        
            characters.append(c)
            
    x_list = read_from_testing_data(test_file)
    output = open(output_filename, "w+")

    character_id = 0
    for x in x_list:
        if len(x) == 0:
            print("Error! Sentecen with length 0!")
            continue
        if len(x) > 1:
            z, _ = model(x)
    #        print(z)
    #        print(z.argmax())
            for (i, prediction) in enumerate(z):
                output.write(characters[character_id])
                if prediction.argmax().item() == 1:
                    output.write("  ")
                character_id += 1
        if character_id < len(characters):
            output.write(characters[character_id])
            character_id += 1
#       print("space: %d %c" % (character_id, characters[character_id]))
        output.write("  ")
        if characters[character_id] == '\n':
#                print("end of line: %d %d %d" % (sentence_id, token_id, character_id)).
            output.write("\n")
            character_id += 1
    torch.cuda.empty_cache()
    output.close()    
    
if __name__ == '__main__':
    net = main('data/bakeoff/training/pku_training.utf8', 
               'data/bakeoff/testing/pku_test.utf8',
               num_sentences = 200, num_epochs=5)
    
