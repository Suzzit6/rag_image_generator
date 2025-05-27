import streamlit as st
import time
import re
import os
from PIL import Image
from main import (
    load_vectorstore,
    initialize_rag_system,
    create_qa_chain,
    load_kandinsky_pipeline,
    generate_kandinsky_image
)

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