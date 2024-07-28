from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader 
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
from langchain_community.embeddings import OllamaEmbeddings
import tempfile
import whisper
from pytube import YouTube
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



if not os.path.exists("transcription.txt"):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()

    # Let's load the base model. This is not the most accurate
    # model but it's fast.
    whisper_model = whisper.load_model("base")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open("transcription.txt", "w") as file:
            file.write(transcription)

with open("transcription.txt") as file:
    transcription = file.read()

loader = TextLoader("transcription.txt")
text_documents = loader.load()
text_documents

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
text_splitter.split_documents(text_documents)

model = Ollama(model="llama3")

parser = StrOutputParser()

template = """
You are an intelligent AI assistent curated to analyze PDF's. Answer the question based on the context provided below. If you can't 
answer the question from the given context, reply "I don't know".

Context: {context}

Question: {question}
"""

# translation_template = """
# Translate {answer} to {language}
# """

prompt = PromptTemplate.from_template(template)
## Adding some extra spice - answering in your local language
# translation_prompt = ChatPromptTemplate.from_template(translation_template)

chain = prompt | model | parser
## information on the chain -> chain.input_schema.schema()

# translation_chain = ({"answer": chain, "language": itemgetter("language")} | translation_prompt | model | parser)

## Initializing the vector store with a suitable embedding -> using a OllamaEmbeddings model for now
vectorStore = DocArrayInMemorySearch.from_documents(pdf_pages, embedding = OllamaEmbeddings(model="llama3"))
ret = vectorStore.as_retriever()

chain = (
    {"context": itemgetter("question") | ret, "question": itemgetter("question")}
    | prompt
    | model 
    | parser
)

chain.invoke("How does transformers work?")