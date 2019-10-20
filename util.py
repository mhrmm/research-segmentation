import torch
import os

def cudaify(x):
    if torch.cuda.is_available():
        cuda = torch.device('cuda:2')
        return x.cuda(cuda)
    else: 
        return x
    
def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    

class Cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.join(os.getcwd(), newPath)

    def __enter__(self):
        if not os.path.exists(self.newPath):
            os.mkdir(self.newPath)
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)