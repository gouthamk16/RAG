import torch
import torch.nn as nn
from tokenizer import tokenize

def embed(text, embedding_dim = 256, save_path = None):

    # Tokenize the text
    tokens = tokenize(text)
    
    # Create a tensor from the tokens
    tensor = torch.tensor(tokens).unsqueeze(0)

    # Define the embedding model
    class Embedder(nn.Module):
        def __init__(self, vocab_size, embed_size):
            super(Embedder, self).__init__()
            self.embed = nn.Embedding(vocab_size, embed_size)
            
        def forward(self, x):
            return self.embed(x)

    embedder = Embedder(vocab_size=max(tokens)+1, embed_size=embedding_dim)
    embedder.eval()

    # Get the embeddings
    with torch.no_grad():
        embeddings = embedder(tensor)

    if save_path:
        torch.save(embeddings, save_path)

    return embeddings