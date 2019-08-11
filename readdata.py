import nltk
from pytorch_transformers import BertTokenizer
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'



def read_from_training_data(filename):
    """
    Reads a file containing tokenized Chinese plaintext. 
    
    Returns a tuple (sents, flags), where sents is a list of sentences (i.e.
    a list of Chinese characters), and flags is a list of word boundaries
    (i.e., a list of Booleans that specify whether the kth character
    terminates a word.)
    
    Normally, sentence breaks are considered to be periods,
    exclamation points, and semicolons. This function will automatically
    insert sentence breaks if a sentence exceeds 300 characters. 
    
    """    
    characters = open(filename).read()
    #x_list is a list of sentences, which means x_list is a 2-d array
    sentence_cnt = 0
    x_list = [[]]
    y_list = [[]]
    alpha_count = 0
    for i in range(len(characters)):
        if not characters[i] == ' ':
            if characters[i] in ALPHABET:
                alpha_count += 1
                
            if characters[i] == '\n':
                if not len(x_list[sentence_cnt]) == 0:
                    sentence_cnt += 1
                    x_list.append([])
                    y_list.append([])
                continue
            
            if len(x_list[sentence_cnt]) >= 300:
                sentence_cnt += 1
                x_list.append([])
                y_list.append([])
            
            y_list[sentence_cnt].append((characters[i + 1] == ' '))
            x_list[sentence_cnt].append(characters[i])
            if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
                sentence_cnt += 1
                x_list.append([])
                y_list.append([])
    print(len(x_list))
    print(alpha_count)
    print(len(characters))
    return zip(x_list, y_list)

 
def read_from_testing_data(filename):
    """
    Reads a file containing untokenized Chinese plaintext. 
    
    Returns a list of sentences (where each sentence is a list of Chinese
    characters).
    
    Normally, sentence breaks are considered to be periods,
    exclamation points, and semicolons.  
    
    """    
    characters = open(filename).read()
    sentence_cnt = 0
    x_list = [[]]
    for i in range(len(characters)):
        if characters[i] == '\n':
            if not len(x_list[sentence_cnt]) == 0:
                sentence_cnt += 1
                x_list.append([])
            continue
        x_list[sentence_cnt].append(characters[i])
        if characters[i] == '。' or characters[i] == '！' or characters[i] == '；':
            sentence_cnt += 1
            x_list.append([])
    x_list = [x for x in x_list if len(x) > 0]
    return x_list

def SentencesReader(filename, file_type):
    """
    Dispatcher function for read_from_training_data and read_from_testing_data.
    
    """
    assert(file_type == 'testing' or file_type == 'training')
    if(file_type == 'testing'):
        return read_from_testing_data(filename)
    if(file_type == 'training'):
        x_list, _ = read_from_training_data(filename)
        return x_list
    
def ReadEnglish(filename):
    characters = open(filename).read()
    split_tool = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = split_tool.tokenize(characters)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case = False)
    i = 0
    x_list = []
    y_list = []
    while i < len(sentences):
        if '#' in sentences[i]:
            del sentences[i]
            continue
        # We just discard sentences that contain names:
        if '.' in sentences[i][:(len(sentences[i]) - 1)]:
            del sentences[i]
            continue
        processed_sentence = ""
        for j in range(len(sentences[i])):
            if sentences[i][j] == ',' or sentences[i][j] == '.' or sentences[i][j] == ':' or sentences[i][j] == ';' or sentences[i][j] == '"':
                processed_sentence = processed_sentence + ' ' + sentences[i][j] + ' '
            else:
                processed_sentence += sentences[i][j]
#        without_space = sentences[i].replace(" ", "")
#        print(without_space)
#        print(nltk.word_tokenize(sentences[i]))
        x_list.append([])
        y_list.append([])
        tokenized_text = tokenizer.tokenize(sentences[i])
        x_list[i] = tokenized_text
#        print(tokenized_text)
        pos = 0
        for j in range(len(tokenized_text) - 1):
            length = len(tokenized_text[j].replace('#', ''))
            pos += length
            if processed_sentence[pos] == ' ' or processed_sentence[pos] == '\n':
                y_list[i].append(True)
                while processed_sentence[pos] == ' ' or processed_sentence[pos] == '\n':
                    pos += 1
            else:
                y_list[i].append(False)
        y_list[i].append(True)
        i += 1
    return x_list, y_list 