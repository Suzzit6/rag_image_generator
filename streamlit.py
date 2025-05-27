import streamlit as st
import time
import re
import os
from PIL import Image
# from main import (
#     load_vectorstore,
#     initialize_rag_system,
#     create_qa_chain,
#     load_kandinsky_pipeline,
#     generate_kandinsky_image
# )
import os
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
from PIL import Image
import os
import time


custom_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=("""You are a humorous storyteller bot.

Answer the following question with:
1. A funny story-based response using ONLY the given context.
2. A separate line: IMAGE_PROMPT: a vivid, descriptive visual prompt of the key character or scene for image generation.

Context: {context}
Question: {question}

Funny Answer:
 """   )
)

def load_documents_from_folder(folder_path):
    """Load all supported documents from a folder"""
    docs = []
    processed_files = []
    
    if not os.path.exists(folder_path):
        return [], []
        
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
            
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            
            
        docs.extend(loader.load())
    
    return docs, processed_files


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    """Split documents into chunks"""
    if not documents:
        return []
        
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    
    return chunks


def embed_and_store(chunks):
    """Create embeddings and store in FAISS vector database"""
    if not chunks:
        return None
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    vectorstore_dir = "rag_vectorstore"
    os.makedirs(vectorstore_dir, exist_ok=True)
    vectorstore.save_local(vectorstore_dir)
    
    return vectorstore


def load_vectorstore():
    """Load a pre-existing vector store"""
    vectorstore_dir = "rag_vectorstore"
    
    if not os.path.exists(vectorstore_dir):
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
        
        return vectorstore
    except Exception as e:
        return None
    
    
def create_qa_chain(vectorstore, api_key=None):
    """Create a question answering chain with the vector store"""
    if vectorstore is None:
        return None
        
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"k": 3, "score_threshold": 0.1}  
    )
    
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
        return None
    
    # # Set the API key in the environment
    # os.environ["GOOGLE_API_KEY"] = api_key
    
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)
        prompt = custom_prompt_template
        
        qa_chain = RetrievalQA.from_chain_type(
          llm=llm,
          chain_type="stuff",
          retriever=retriever,
          chain_type_kwargs={"prompt": prompt},
          return_source_documents=True
)
        
        return qa_chain
    except Exception as e:
        return None

def initialize_rag_system(folder_path, chunk_size=500, chunk_overlap=100):
    """Initialize the RAG system by loading documents, chunking, embedding, and storing"""
    documents, processed_files = load_documents_from_folder(folder_path)
    
    if not documents:
        return None, None
    
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    
    vectorstore = embed_and_store(chunks)
    
    return vectorstore, processed_files





def load_kandinsky_pipeline():
    """Load both prior and decoder pipelines for Kandinsky 2.2"""
    try:
        # Load the prior pipeline (for text embeddings)
        prior_model_id = "kandinsky-community/kandinsky-2-2-prior"
        prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
            prior_model_id,
            torch_dtype=torch.float16
        )

        # Load the decoder pipeline (for image generation)
        decoder_model_id = "kandinsky-community/kandinsky-2-2-decoder"
        decoder_pipe = KandinskyV22Pipeline.from_pretrained(
            decoder_model_id,
            torch_dtype=torch.float16
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        prior_pipe = prior_pipe.to(device)
        decoder_pipe = decoder_pipe.to(device)

        print(f"Kandinsky pipelines loaded successfully on {device}")
        return prior_pipe, decoder_pipe

    except Exception as e:
        print(f"Error loading Kandinsky pipelines: {e}")
        return None, None

def generate_kandinsky_image(prompt, output_path="generated_image.png", pipelines=None):
    """Generate image using Kandinsky 2.2"""
    try:
        if pipelines is None:
            prior_pipe, decoder_pipe = load_kandinsky_pipeline()
        else:
            prior_pipe, decoder_pipe = pipelines

        if prior_pipe is None or decoder_pipe is None:
            print("Failed to load pipelines")
            return None

        print(f"Generating image for prompt: {prompt}")

        # Define negative prompt
        negative_prompt = "low quality, bad quality, blurry, pixelated, distorted"

        # Generate image and negative image embeddings using prior pipeline
        image_embeds, negative_image_embeds = prior_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=1.0,
            num_inference_steps=25
        ).to_tuple()

        # Generate image using decoder pipeline
        image = decoder_pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=768,
            width=768,
            num_inference_steps=50,
            guidance_scale=4.0
        ).images[0]

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Save the image
        image.save(output_path)
        print(f"Image saved successfully at: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def test_kandinsky():
    """Test function to verify Kandinsky is working"""
    test_prompt = "A beautiful sunset over mountains, digital art"
    pipelines = load_kandinsky_pipeline()

    if pipelines[0] is None or pipelines[1] is None:
        print("Failed to load pipelines for testing")
        return

    result = generate_kandinsky_image(test_prompt, "test_output.png", pipelines)
    if result:
        print("Test successful!")
    else:
        print("Test failed!")



# Set page config
st.set_page_config(
    page_title="Humorous Storyteller Bot",
    page_icon="📚",
    layout="wide"
)

# App title and description
st.title("📚 Humorous Storyteller Bot")
st.markdown("""
This app uses RAG (Retrieval Augmented Generation) to create funny stories based on your questions,
and generates accompanying images using Kandinsky AI.
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    folder_path = st.text_input("Story Books Folder Path", "story_books")
    chunk_size = st.slider("Chunk Size", 100, 1000, 500)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 100)
    
    # Initialize/Load button
    if st.button("Initialize/Reload RAG System"):
        with st.spinner("Initializing RAG system..."):
            vectorstore = load_vectorstore()
            if vectorstore is None:
                vectorstore, processed_files = initialize_rag_system(folder_path, chunk_size, chunk_overlap)
                if vectorstore:
                    st.success(f"Successfully built vector store from {folder_path}")
                else:
                    st.error(f"Failed to initialize RAG system from {folder_path}")
            else:
                st.success("Loaded existing vector store")
            
            st.session_state.vectorstore = vectorstore
            
            # Initialize Kandinsky pipelines
            with st.spinner("Loading image generation models..."):
                pipelines = load_kandinsky_pipeline()
                if pipelines[0] is not None and pipelines[1] is not None:
                    st.success("Image generation models loaded successfully")
                    st.session_state.pipelines = pipelines
                else:
                    st.error("Failed to load image generation models")
    
    # Show some info about the models
    st.markdown("### About")
    st.markdown("""
    - Text Generation: Google Gemini 2.0 Flash
    - Embeddings: all-MiniLM-L6-v2
    - Image Generation: Kandinsky 2.2
    """)

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
    
if 'pipelines' not in st.session_state:
    st.session_state.pipelines = None
    
if 'story_response' not in st.session_state:
    st.session_state.story_response = ""
    
if 'image_path' not in st.session_state:
    st.session_state.image_path = None

if 'query_submitted' not in st.session_state:
    st.session_state.query_submitted = False

# Main interface
query = st.text_input("Ask a question:", key="query_input")

# Process the query
if st.button("Generate Story & Image") or st.session_state.query_submitted:
    st.session_state.query_submitted = True
    
    if not query:
        st.warning("Please enter a question")
    elif st.session_state.vectorstore is None:
        st.warning("Please initialize the RAG system first")
    else:
        # Create columns for the response and image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Process the query and get the story response
            with st.spinner("Generating story..."):
                qa_chain = create_qa_chain(st.session_state.vectorstore)
                if qa_chain:
                    try:
                        response = qa_chain.invoke({"query": query})
                        st.session_state.story_response = response['result']
                        
                        # Display the story (without the image prompt)
                        story_text = re.sub(r"IMAGE_PROMPT:.*", "", st.session_state.story_response)
                        st.markdown("### Your Funny Story")
                        st.markdown(story_text)
                        
                    except Exception as e:
                        st.error(f"Error generating story: {str(e)}")
                else:
                    st.error("Failed to create QA chain")
        
        with col2:
            # Extract image prompt and generate image
            with st.spinner("Generating image..."):
                image_prompt_match = re.search(r"IMAGE_PROMPT:(.*)", st.session_state.story_response)
                
                if image_prompt_match:
                    image_prompt = image_prompt_match.group(1).strip()
                    st.markdown("### Image Prompt")
                    st.write(image_prompt)
                    
                    # Load pipelines if not already loaded
                    if st.session_state.pipelines is None:
                        st.session_state.pipelines = load_kandinsky_pipeline()
                    
                    if st.session_state.pipelines[0] is not None and st.session_state.pipelines[1] is not None:
                        # Generate a unique filename for each image
                        timestamp = int(time.time())
                        output_dir = "generated_images"
                        os.makedirs(output_dir, exist_ok=True)
                        image_path = f"{output_dir}/story_image_{timestamp}.png"
                        
                        # Generate the image
                        result_path = generate_kandinsky_image(
                            image_prompt, 
                            image_path, 
                            st.session_state.pipelines
                        )
                        
                        if result_path:
                            st.session_state.image_path = result_path
                            st.markdown("### Generated Image")
                            st.image(result_path)
                        else:
                            st.error("Failed to generate image")
                    else:
                        st.error("Image generation models are not loaded")
                else:
                    st.warning("No image prompt found in the response")

# Display saved image if available
if 'image_path' in st.session_state and st.session_state.image_path:
    if not st.session_state.query_submitted:  # Only show if not already shown above
        st.markdown("### Previously Generated Image")
        st.image(st.session_state.image_path)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit, LangChain, and Kandinsky")