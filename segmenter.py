from readdata import read_test_data

class XE:
    X = 0
    E = 1
    
    def encode(self, line):
        sent = []
        flags = []
        line = line.strip() + " "
        for i in range(len(line)):
            if not line[i] == ' ':
                if line[i + 1] == ' ':
                    flags.append(XE.E)
                else:
                    flags.append(XE.X)
                sent.append(line[i])
        return sent, flags        

    def decode(self, tokens, flags):
        result = ""
        for tok, flag in zip(tokens, flags):
            result += tok
            if flag == XE.E:
                result += "  "
        return result.strip()
    
    def domain_size(self):
        return 2
    
    def terminator(self):
        return XE.E

    
class BMES:
    B = 0
    M = 1
    E = 2
    S = 3

    def encode(self, line):
        sent = []
        flags = []
        line = line.strip() + " "
        starting_new_word = True
        for i in range(len(line)):
            if not line[i] == ' ':
                if line[i + 1] == ' ':
                    if starting_new_word:
                        flags.append(BMES.S)
                    else:
                        flags.append(BMES.E)
                    starting_new_word = True
                else:
                    if starting_new_word:
                        flags.append(BMES.B)
                    else:
                        flags.append(BMES.M)
                    starting_new_word = False
                sent.append(line[i])
        return sent, flags        

    def decode(self, tokens, flags):
        result = ""
        for tok, flag in zip(tokens, flags):
            result += tok
            if flag == BMES.E or flag == BMES.S:
                result += "  "
        return result.strip()
    
    def domain_size(self):
        return 4

    def terminator(self):
        return XE.E



def segment_sent(model, sent):
    if len(sent) <= 1:
        return sent
    else:
        segmenter = model.segmenter()
        flags = segmenter(sent)
        tokens = [tok for tok in sent]
        assert len(tokens) == len(flags)
        return model.encoding.decode(tokens, flags)
       
def segment_sents(model, lines):
    result = ""
    for sents in lines:
        for sent in sents:
            result += segment_sent(model, sent) + "  "
        result = result.strip() + "\n"
    return result.strip()

def segment_file(model, input_file, output_file):
    with open(input_file) as inhandle:
        with open(output_file, 'w') as outhandle:
            lines = read_test_data(inhandle)
            outhandle.write(segment_sents(model, lines))
    

            