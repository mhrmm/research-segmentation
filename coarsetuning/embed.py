import torch
from pytorch_transformers import BertTokenizer, BertModel
import pandas as pd

#MODEL_NAME = 'bert-base-chinese' 
MODEL_NAME = 'bert-base-multilingual-cased'

if MODEL_NAME == 'bert-base-chinese':
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    bert_model.eval()
else:
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case = False)
    bert_model = BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    bert_model.eval()
  
def process_dataset(dataset, csvfile, embedder):
    """
    Converts Chinese segmentation data into training vectors and writes them
    to a CSV file, in which the first column is the category.
    
    """
    with open(csvfile, 'w') as outhandle:
        for i, (tokens, flags) in enumerate(dataset):            
            if i % 100 == 0:
                print('processed {} sentences'.format(i))
            if len(tokens) > 0:
                vec = embedder(tokens, flags)
                df = pd.DataFrame(vec.detach().numpy())
                outhandle.write(df.to_csv(header=False, index=False))    
        
class Embedder:

    def __call__(self, tokens, flags):
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        bert_output = bert_model(tokens_tensor)
        return self.process_bert_output(bert_output, flags).detach()

    def process_bert_output(self, bert_output, flags):
        raise NotImplementedError("Cannot call a generic Embedder.")
  
class SimpleEmbedder(Embedder):
         
    def process_bert_output(self, bert_output, flags): 
        embeddings = bert_output[0].squeeze(0)
        labels = torch.tensor(flags).unsqueeze(1).float()
        return torch.cat([labels, embeddings], dim=1)

class GapEmbedder(Embedder):
        
    def process_bert_output(self, bert_output, flags): 
        embeddings = bert_output[0].squeeze(0)
        gap_embeddings = torch.cat([embeddings[:-1], embeddings[1:]], dim=1)
        labels = torch.tensor(flags)[:-1].unsqueeze(1).float()
        return torch.cat([labels, gap_embeddings], dim=1)
          

class GapEmbedderAverageOfFirstAndLastLayers(Embedder):
        
    def process_bert_output(self, bert_output, flags):
        first_layer = bert_output[2][0].squeeze(0)
        last_layer = bert_output[2][12].squeeze(0)
        embeddings = (first_layer + last_layer) / 2.
        gap_embeddings = torch.cat([embeddings[:-1], embeddings[1:]], dim=1)
        labels = torch.tensor(flags)[:-1].unsqueeze(1).float()
        return torch.cat([labels, gap_embeddings], dim=1)      




