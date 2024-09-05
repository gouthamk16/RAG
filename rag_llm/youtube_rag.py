from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from torch import cuda
from langchain.vectorstores import Chroma


## Loading the youtube video transcript into a text file
YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=IARh8shW-Ww"

def getVideoId(url):
    if url.startswith("https"):
        id = url[32:len(url)]
    elif url.startswith("www"):
        id = url[24:len(url)]
    return id

def get_transcript(video_id):
    try:
        # Retrieve the transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine the transcript into a single string
        transcript_text = ' '.join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        return str(e)
    
def youtube_rag(question, video_url):
    video_id = getVideoId(video_url)
    transcript = get_transcript(video_id)

    # Text Splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splitted_text = text_splitter.split_text(transcript)

    # Set up the model and vector store
    model = Ollama(model="llama3")
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = DocArrayInMemorySearch.from_texts(splitted_text, embeddings)
    retriever = vectorstore.as_retriever()

    # Define the template and chain
    template = """
    You are an intelligent AI assistant who summrizes and analyzes transcripts of youtube videos. Answer the question based on the context provided below. If you can't answer the question from the context, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    # prompt = PromptTemplate.from_template(template)
    # parser = StrOutputParser()

    # # Create the chain
    # chain = (
    #     {"context": retriever, "question": question}
    #     | prompt
    #     | model
    #     | parser
    # )

    prompt = PromptTemplate.from_template(template)
    parser = StrOutputParser()

    # Langchain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model 
        | parser
    )

    # Get the response
    return chain.invoke(question)
    
# video_id = getVideoId(YOUTUBE_VIDEO)
# print("Video id: ", video_id)

# transcript = get_transcript(video_id)

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False,
# )

# splitted_text = text_splitter.split_text(transcript)

# # Defining the model, embedding and the parser
# model = Ollama(model="llama3")
# embeddings = OllamaEmbeddings(model = "llama3")
# parser = StrOutputParser()

# ## Trying out huggingface embedding model along wih chroma db
# ## TO-DO: Test this implementation
# # embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
# # device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
# # embedding_model = HuggingFaceEmbeddings(
# #     model_name=embed_model_id,
# #     model_kwargs={'device': device},
# #     encode_kwargs={'device': device, 'batch_size': 32}
# # )
# # loader = TextLoader(transcript)
# # data = loader.load()
# # text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=0)
# # all_splits = text_splitter.split_documents(data)
# # vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

# ## Setting up the vectorstore
# vectorstore = DocArrayInMemorySearch.from_texts(splitted_text, embeddings)
# ret = vectorstore.as_retriever()

# ## Defining the finetuning prompt
# template = """
# You are an intelligent AI assistent curated to analyze PDF's. Answer the question based on the context provided below. Just the answer, no additional messages. If you can't 
# answer the question from the given context, reply "I don't know".

# Context: {context}

# Question: {question}
# """

# prompt = PromptTemplate.from_template(template)


# # Langchain
# chain = (
#     {"context": ret, "question": RunnablePassthrough()}
#     | prompt
#     | model 
#     | parser
# )

# # Invoking the chain from a query
# response = chain.invoke("What is neom")

# print(response)