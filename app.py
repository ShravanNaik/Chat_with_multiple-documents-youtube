import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import requests
from bs4 import BeautifulSoup
 
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
 
def get_pdf_text(pdf_docs):
     text=""
     for pdf in pdf_docs:
         pdf_reader= PdfReader(pdf)
         for page in pdf_reader.pages:
             text+= page.extract_text()
     return  text
 
def get_youtbe_transcript(link):
     try:
         video_id = link.split("v=")[1]
         transcript = YouTubeTranscriptApi.get_transcript(video_id)
         text = " ".join(t['text'] for t in transcript)
         return text
     except Exception as e:
         st.error(f"Error fetching trascript: {e}")
         return ""
 # Function to extract text from a website
def get_website_text(url):
     try:
         response = requests.get(url)
         soup = BeautifulSoup(response.content, "html.parser")
         text = soup.get_text()
         return text
     except Exception as e:
         st.error(f"Error fetching website content: {e}")
         return ""
 
def get_text_chunks(text):
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
     chunks = text_splitter.split_text(text)
     return chunks
 
 
def get_vector_store(text_chunks,embeddings):
     try:
         vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
         vector_store.save_local("faiss_index")
     except Exception as e:
         st.error("Error Creating Index:{e}")
 
 
def create_vector_store(text_chunks):
     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
     return vector_store
 
def get_conversational_chain():
 
     prompt_template = """
     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
     Context:\n {context}?\n
     Question: \n{question}\n
 
     Answer:
     """
 
     model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                              temperature=0.3)
 
     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
     return chain
 
 
 
def user_input(user_question,vector_store):
     docs = vector_store.similarity_search(user_question)
     chain = get_conversational_chain()
     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
     return response["output_text"]
 
 
 
def main():
     st.set_page_config("Chat with Files, Videos or Websites",layout="wide")
     st.markdown("""
         <style>
             .main {
                 background-color: #f5f5f5;
             }
             .header {
                 color: #ff6347;
                 font-size: 36px;
                 font-weight: bold;
                 text-align: center;
             }
             .sidebar .sidebar-content {
                 background-color: #ff6347;
             }
             .sidebar .sidebar-content .widget {
                 color: white;
             }
             .btn-primary {
                 background-color: #ff6347;
                 color: white;
             }
             .stTextInput>div>div>input {
                 background-color: #f1f1f1;
                 color: #333;
             }
             .stTextInput>div>label {
                 color: #ff6347;
             }
             .stButton>button {
                 background-color: #4CAF50;
                 color: white;
             }
         </style>
     """, unsafe_allow_html=True)
 
 
     st.markdown('<div class="header">Chat with PDF, YouTube, or Websites 💬</div>', unsafe_allow_html=True)
     st.header("Chat with PDF ,Youtube or Websites")
 
 
     with st.sidebar:
         st.title("Enter Your Google API Key:")
         user_api_key = st.text_input("Google API Key", type="password", help="Enter your Google API key for the app.")
         if user_api_key:
             genai.configure(api_key=user_api_key)
             st.success("API Key SUccessfully configured fot the session")
 
         st.title("Upload Data:")
         pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
         youtube_link = st.text_input("Enter your youtube link")
         website_link = st.text_input("Enter website link")
         if user_api_key:
 
             if st.button("Submit & Process"):
                 with st.spinner("Processing..."):
                     combined_text =""
 
                     if pdf_docs:
                         combined_text +=get_pdf_text(pdf_docs)
 
                     if youtube_link:
                         combined_text += get_youtbe_transcript(youtube_link)
 
                     if website_link:
                         combined_text += get_website_text(website_link)
 
                     if combined_text:
                         text_chunks = get_text_chunks(combined_text)
                         vector_store = create_vector_store(text_chunks)
                         st.session_state["vector_store"] = vector_store
                         st.success("Done")
     user_question = st.text_input("Ask a question based on the processed data")
     if user_question and "vector_store" in st.session_state:
         vector_store = st.session_state["vector_store"]
         with st.spinner("Fetching answer..."):
             answer = user_input(user_question, vector_store)
             st.write("Reply:", answer)
     elif user_question:
         st.error("Please process some data first!")
 
 
 
 if __name__ == "__main__":
     main()
