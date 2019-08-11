import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
class DropoutClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 200):
        super(DropoutClassifier, self).__init__()
        self.dropout1 = torch.nn.Dropout(p=0.2)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        #self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        self.linear3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_vec):
        nextout = input_vec
        nextout = self.dropout1(nextout)
        nextout = self.linear1(nextout)
        nextout = nextout.clamp(min=0)
        #nextout = self.linear2(nextout).clamp(min=0)
        nextout = self.dropout2(nextout)    
        nextout = self.linear3(nextout)
        return nextout
    
 
