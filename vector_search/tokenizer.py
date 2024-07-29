import tiktoken

def tokenize(text, llm_tokenizer="gpt-4"):
    # Tokenizer
    enc = tiktoken.encoding_for_model(llm_tokenizer)

    # Tokenize the text
    tokens = enc.encode(text)
    
    return tokens