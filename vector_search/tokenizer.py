# import tiktoken

# def tokenize(text, llm_tokenizer="gpt-4"):
#     # Tokenizer
#     enc = tiktoken.encoding_for_model(llm_tokenizer)

#     # Tokenize the text
#     tokens = enc.encode(text)
    
#     return tokens


import tiktoken

def tokenize(text, llm_tokenizer="gpt-4", max_length=10, padding_token=0):
    """
    Tokenizes the input text using tiktoken, and pads or truncates the token sequence to a fixed length.

    Args:
        text (str): The input text to tokenize.
        llm_tokenizer (str): The name of the tokenizer model (default is "gpt-4").
        max_length (int): The desired length for each token sequence (default is 10).
        padding_token (int): The token to use for padding if the sequence is shorter than max_length (default is 0).

    Returns:
        List[int]: A list of tokens of length `max_length`.
    """
    # Tokenizer
    enc = tiktoken.encoding_for_model(llm_tokenizer)

    # Tokenize the text
    tokens = enc.encode(text)

    # Adjust the length of tokens to max_length
    if len(tokens) > max_length:
        # Truncate if the sequence is longer than max_length
        tokens = tokens[:max_length]
    else:
        # Pad with the padding_token if the sequence is shorter than max_length
        tokens += [padding_token] * (max_length - len(tokens))

    return tokens
