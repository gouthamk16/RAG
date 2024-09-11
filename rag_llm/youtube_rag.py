from langchain_community.llms import Ollama 
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
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

    # Create documents from the splitted text
    docs = text_splitter.create_documents(splitted_text)

    # Set up the model, embeddings and vector store
    model = Ollama(model="llama3")

    embedding_model = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Define the template and chain
    template = """
    You are an intelligent AI assistant who summrizes and analyzes transcripts of youtube videos. Answer the question based on the context provided below. If you can't answer the question from the context, reply "I don't know".

    Context: {context}

    Question: {question}
    """

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


# # Sample usage 
# url = "https://www.youtube.com/watch?v=UcJXHsGiUVg"
# question = "what does grace hayden enjoy?"

# print(youtube_rag(question, url))
    
