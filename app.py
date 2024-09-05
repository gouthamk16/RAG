## STREAMLIT APP WILL BE CREATED HERE !!
import streamlit as st
from rag_llm.youtube_rag import youtube_rag
from rag_llm.pdf_rag import pdf_rag
from rag_llm.web_rag import web_rag

st.title("AI Model Summarizer")

# User selects model
option = st.selectbox(
    "Choose a model for summarization:",
    ("YouTube Video Summarization", "PDF Summarization", "Website Summarization")
)

# User inputs question
question = st.text_input("Enter your question:")

# Model-specific inputs and results
if option == "YouTube Video Summarization":
    video_url = st.text_input("Enter YouTube video URL:")
    if st.button("Summarize Video"):
        if video_url and question:
            response = youtube_rag(question, video_url)
            st.write(response)

elif option == "PDF Summarization":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Summarize PDF"):
        if pdf_file and question:
            response = pdf_rag(question, pdf_file)
            st.write(response)

elif option == "Website Summarization":
    url = st.text_input("Enter website URL:")
    if st.button("Summarize Website"):
        if url and question:
            response = web_rag(question, url)
            st.write(response)
