import numpy as np

class vectorStore:
    def __init__(self):
        self.vector = {} # Dictionary of vectors
        self.index = {}
    
    def similarity(vector_a, vector_b, method="cosine"):
        if method == "cosine":
            return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        else:
            raise ValueError("Invalid similarity method")

    def add_vectors(self, vector_id, vector):
        self.vector[vector_id] = vector
        self.update_index(vector_id, vector)
    
    def get_vector(self, vector_id):
        return self.vector.get(vector_id)

    def update_index(self, vector_id, vector):
        for existing_id, exsiting_vector in self.vector.items():
            # Calculating cosine similarity between the vectors to determine the new index
            cosine_similarity = self.similarity(vector, exsiting_vector)
            if existing_id not in self.index:
                self.index[existing_id] = {}
            self.index[existing_id][vector_id] = cosine_similarity

    def get_similar(self, query, top_n=5):
        results = []
        for existing_id, existing_vector in self.vector.items():
            cosine_similarity = self.similarity(query, existing_vector)
            results.append((existing_id, cosine_similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_n]