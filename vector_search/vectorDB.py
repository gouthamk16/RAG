## Creating a vector database given embeddings 
from embedding import embed
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class VectorDB:
    def __init__(self, embedding_dim = 256):
        self.embedding_dim = embedding_dim
        self.vectors = {}
        self.index = {}
    
    def from_text(self, id, text):
        embedding = embed(text, embedding_dim=self.embedding_dim)
        self.vectors[id] = embedding
        self.index[text] = id
    
    def retrieve(self, query, k=3):
        embedded_query = embed(query, embedding_dim=self.embedding_dim)
        list_of_embeddings = list(self.vectors.values())
        list_of_ids = list(self.vectors.keys())

        similarity = cosine_similarity([embedded_query], list_of_embeddings)[0]

        nearest = similarity.argsort()[::-1][:k]
        nearest_neighbors = [(list_of_ids[i], similarity[i]) for i in nearest]

        return nearest_neighbors



# Create a vector database
vector_db = VectorDB()
vector_db.from_text(1, "Iam Goutham")
vector_db.from_text(2, "Machine Learning is an AI")
vector_db.from_text(3, "I drive a HONDA")

query = "HONDA is a car"

nearest_neighbors = vector_db.retrieve(query, k=3)
print("Nearest neighbors to the query:")
for neighbor in nearest_neighbors:
    print(f"ID: {neighbor[0]}, Similarity: {neighbor[1]}")