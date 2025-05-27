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
    
    
def create_qa_chain(vectorstore, api_key="AIzaSyBE6um0ubRbJb0S_yosy7GcBAouv87yiiI"):
    """Create a question answering chain with the vector store"""
    if vectorstore is None:
        return None
        
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"k": 3, "score_threshold": 0.1}  
    )
    
    if not api_key:
        api_key = "AIzaSyBE6um0ubRbJb0S_yosy7GcBAouv87yiiI"
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


# test_kandinsky()

def main():
    folder_path = "story_books"
    chunk_size = 500
    chunk_overlap = 100

    # Initialize the RAG system
    vectorstore = load_vectorstore()
    if vectorstore is None:
        print("No pre-existing vector store found. Initializing RAG system...")
        vectorstore, processed_files = initialize_rag_system(folder_path, chunk_size, chunk_overlap)
        print(f"Processed {len(processed_files)} files from {folder_path}.")


    # Create the QA chain
    qa_chain = create_qa_chain(vectorstore)

    if qa_chain is None:
        print("Failed to create QA chain.")
        return

    # Example query
    query = "with whom alice was the whole time ?"
    response = qa_chain.invoke({"query": query})
    # sample_response = {
    #     "result": "Alice is a curious young girl who falls down a rabbit hole. IMAGE_PROMPT: Alice in Wonderland, young girl with blonde hair, blue dress, falling down rabbit hole, whimsical fantasy art",
    #     "source_documents": []
    # }

    # print("sample_response:", sample_response['result'])
    print("Response:", response['result'])
    import re
    image_prompt_match = re.search(r"IMAGE_PROMPT:(.*)", response["result"])
    if image_prompt_match:
        image_prompt = image_prompt_match.group(1).strip()
        print("Image Prompt:", image_prompt)

        # Load pipelines once
        pipelines = load_kandinsky_pipeline()

        if pipelines[0] is not None and pipelines[1] is not None:
            image_path = generate_kandinsky_image(image_prompt, "alice_output.png", pipelines)
            if image_path:
                print("Image saved at:", image_path)
            else:
                print("Failed to generate image")
        else:
            print("Failed to load Kandinsky pipelines")
    else:
        print("No image prompt found.")


    print("Source Documents:", [doc.metadata for doc in response['source_documents']])
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")