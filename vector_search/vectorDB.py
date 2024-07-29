## Creating a vector database given embeddings 
from embedding import embed
import torch

class VectorDB:
    def __init__(self, embedding_dim = 256):
        self.embedding_dim = embedding_dim
        self.vector = {}
        self.index = {}
    
    def add_text(self, vectors):
        vector = embed(text, embedding_dim = self.embedding_dim)
        self.vectors.append(vector)
        self.texts.append(text)
        self.text_to_vector[text] = vector
    
    def search(self, query, k = 1):
        query_vector = embed(query, embedding_dim = self.embedding_dim)
        distances = []
        for vector in self.vectors:
            print(query_vector.shape, vector.shape)
            distance = torch.dist(query_vector, vector)
            distances.append(distance)
        distances = torch.tensor(distances)
        indices = torch.argsort(distances)
        return [self.texts[i] for i in indices[:k]]

# Create a vector database
vector_db = VectorDB()


print(results)