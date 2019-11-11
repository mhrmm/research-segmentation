import torch

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
            #mean = layers[12][0][i + w]
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

