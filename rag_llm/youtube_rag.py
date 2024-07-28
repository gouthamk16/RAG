from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_community.embeddings import OllamaEmbeddings
import tempfile
import whisper
from pytube import YouTube
import os
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

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
    
video_id = getVideoId(YOUTUBE_VIDEO)
print("Video id: ", video_id)

transcript = get_transcript(video_id)

# if not os.path.exists("transcription.txt"):
#     youtube = YouTube(YOUTUBE_VIDEO)
#     audio = youtube.streams.filter(only_audio=True).first()

#     # Let's load the base model. This is not the most accurate
#     # model but it's fast.
#     whisper_model = whisper.load_model("base")

#     with tempfile.TemporaryDirectory() as tmpdir:
#         file = audio.download(output_path=tmpdir)
#         transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

#         with open("transcription.txt", "w") as file:
#             file.write(transcription)

# with open("transcription.txt") as file:
#     transcription = file.read()

# loader = TextLoader(transcript)
# transcript = loader.load()

# print("Extracted Transcript: ")
# print(transcript)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# texts = text_splitter.create_documents([transcript])

splitted_text = text_splitter.split_text(transcript)


# # Splitting the transcript into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
# processed_transcript = text_splitter.split_documents(transcript)

# Defining the model, embedding and the parser
model = Ollama(model="llama3")
embeddings = OllamaEmbeddings(model = "llama3")
parser = StrOutputParser()

## Setting up the vectorstore
vectorstore = DocArrayInMemorySearch.from_texts(splitted_text, embeddings)
ret = vectorstore.as_retriever()

## Defining the finetuning prompt
template = """
You are an intelligent AI assistent curated to analyze PDF's. Answer the question based on the context provided below. If you can't 
answer the question from the given context, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)


# Langchain
chain = (
    {"context": ret, "question": RunnablePassthrough()}
    | prompt
    | model 
    | parser
)

# Invoking the chain from a query
response = chain.invoke("What is neom")

print(response)