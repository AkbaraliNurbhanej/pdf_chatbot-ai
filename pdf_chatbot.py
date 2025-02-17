import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Together
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import time
import re
from urllib.parse import urlparse, parse_qs

# Ensure that the necessary environment variables are loaded
load_dotenv()

# Check for required API keys
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set in the environment variables.")
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set in the environment variables.")
if not os.getenv("TOGETHER_API_KEY"):
    st.error("TOGETHER_API_KEY is not set in the environment variables.")

# Load environment variables
load_dotenv()

# API keys setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Multi-Source Chat",
    page_icon="🤖",
    layout="wide"
)


# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = False
if "source_type" not in st.session_state:
    st.session_state.source_type = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "source_data" not in st.session_state:
    st.session_state.source_data = {}

def get_conversation_chain(vectorstore, llm_model):
    """Create conversation chain based on selected LLM model."""
    try:
        if llm_model == "Gemini Pro":
            llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro")
        elif llm_model == "Mixtral 8x7B":
            llm = Together(model="mistralai/Mistral-7B-Instruct-v0.3")
        elif llm_model == "LLaMA 2 70B":
            llm = Together(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
        elif llm_model == "Qwen 32B":
            llm = Together(model="Qwen/Qwen2.5-Coder-32B-Instruct")
        elif llm_model == "Mistral 7B":
            llm = Together(model="mistralai/Mistral-7B")
        elif llm_model == "Mistral 13B":
            llm = Together(model="mistralai/Mistral-13B")
        elif llm_model == "LLaMA 2 13B":
            llm = Together(model="meta-llama/Llama-2-13B")
        else:
            st.error("Invalid LLM model selected.")
            return None
        
        # Create the conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )
        
        return conversation_chain

    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    try:
        if 'youtube.com' in url:
            query = urlparse(url).query
            return parse_qs(query)['v'][0]
        elif 'youtu.be' in url:
            return url.split('/')[-1]
    except:
        return None
    return None

def get_youtube_transcript(url):
    """Get YouTube transcript using youtube_transcript_api directly"""
    try:
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL")
            return None
            
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text = ' '.join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        st.error(f"Error getting YouTube transcript: {str(e)}")
        return None

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text() or ""
                text += extracted_text.encode('utf-8', errors='ignore').decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def get_web_content(url):
    """Get web content using WebBaseLoader"""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return ' '.join([doc.page_content for doc in data])
    except Exception as e:
        st.error(f"Error processing website: {str(e)}")
        return None
    
def get_document_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {str(e)}")
        return None

def create_vectorstore(chunks, embeddings=None):
    """Create FAISS vectorstore for all content types"""
    try:
        if embeddings is None:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {str(e)}")
        return None

def get_conversation_chain(vectorstore):
    try:
        llm = ChatGoogleGenerativeAI(
            temperature=0.7,
            model='gemini-1.5-flash',
            convert_system_message_to_human=True
        )
        
        template = """You are a helpful AI assistant that helps users understand content from various sources.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always maintain a professional and helpful tone.
        
        Context: {context}
        
        Question: {question}
        Helpful Answer:"""

        prompt = PromptTemplate(
            input_variables=['context', 'question'],
            template=template
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt},
            return_source_documents=True
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

def process_content(content, source_type, source_identifier):
    """Process content and store source information"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"Processing {source_type}...")
        progress_bar.progress(20)
        
        if not content:
            return False
            
        status_text.text("Splitting content into chunks...")
        progress_bar.progress(40)
        chunks = get_document_chunks(content)
        if not chunks:
            return False
            
        status_text.text("Creating embeddings and vectorstore...")
        progress_bar.progress(60)
        
        embeddings = (OpenAIEmbeddings() if st.session_state.get('embeddings_type') == "OpenAI" 
                     else GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
        
        vectorstore = create_vectorstore(chunks, embeddings)
        if not vectorstore:
            return False
            
        st.session_state.vectorstore = vectorstore
        
        # Store source data
        st.session_state.source_data[source_type] = {
            'identifier': source_identifier,
            'vectorstore': vectorstore,
            'content': content
        }
        
        status_text.text("Setting up conversation chain...")
        progress_bar.progress(90)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        st.session_state.processComplete = True
        st.session_state.source_type = source_type
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return True
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return False

# Sidebar for source selection and content upload
with st.sidebar:
    st.title("📚 Source Selection")
    source_type = st.selectbox(
        "Choose your content source",
        ["PDF Documents", "Website URL", "YouTube Video"]
    )

    llm_model = st.selectbox(
    "Choose your LLM model",
    ["Gemini Pro", "Mixtral 8x7B", "LLaMA 2 70B", "Qwen 32B"],
    help="Select the LLM model to use for processing"
    )
    
    
    if source_type == "PDF Documents":
        uploaded_files = st.file_uploader(
            "Upload your PDFs",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze"
        )
        
        if uploaded_files:
            file_names = [file.name for file in uploaded_files]
            st.write("📑 Uploaded files:")
            for name in file_names:
                st.write(f"- {name}")
            
            process_button = st.button("Process PDFs", disabled=not uploaded_files)
            if process_button:
                content = get_pdf_text(uploaded_files)
                success = process_content(content, "PDF", ', '.join(file_names))
                if success:
                    st.success("✅ Documents processed successfully!")
                    
    elif source_type == "Website URL":
        url = st.text_input("Enter website URL", key="web_url")
        process_button = st.button("Process Website", disabled=not url)
        if process_button and url:
            content = get_web_content(url)
            success = process_content(content, "Website", url)
            if success:
                st.success("✅ Website processed successfully!")
                
    elif source_type == "YouTube Video":
        url = st.text_input("Enter YouTube URL", key="youtube_url")
        process_button = st.button("Process Video", disabled=not url)
        if process_button and url:
            content = get_youtube_transcript(url)
            success = process_content(content, "YouTube", url)
            if success:
                st.success("✅ Video processed successfully!")


# Main chat interface
st.title("🤖 Multi-Source Chat Assistant")

if not st.session_state.processComplete:
    st.info("👈 Please select a source and process content to begin chatting!")
else:
    st.caption(f"Currently chatting with: {st.session_state.source_type} content")
    
    chat_container = st.container()
    
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
    
    user_question = st.chat_input("Ask a question about your content:")
    
    if user_question:
        with st.chat_message("You"):
            st.write(user_question)
        
        with st.chat_message("Bot"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.conversation({
                        "question": user_question
                    })
                    bot_response = response["answer"]
                    st.write(bot_response)
                    
                    st.session_state.chat_history.append(("You", user_question))
                    st.session_state.chat_history.append(("Bot", bot_response))
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")