
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
from kaggle_secrets import UserSecretsClient
import gradio as gr
import torch
from PIL import Image
import os
import time
import re
from typing import Tuple, Optional, List
import gc# Audio processing imports
try:
    import speech_recognition as sr
    import pyttsx3
    import librosa
    AUDIO_AVAILABLE = True
    print("Audio Available")
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio libraries not available. Install: pip install SpeechRecognition pyttsx3 librosa")

# Global variables to store loaded pipelines and vectorstore
kandinsky_pipelines = None
current_vectorstore = None
qa_chain = None
tts_engine = None
secret_label = "GEMINI_API_KEY"
secret_value = os.getenv(secret_label, None)


# Memory management functions
def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def optimize_pytorch_memory():
    """Set PyTorch memory optimization flags"""
    if torch.cuda.is_available():
        # Enable memory pool fragmentation reduction
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # Set memory fraction to use less GPU memory
        torch.cuda.set_per_process_memory_fraction(0.8)
        
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
        elif filename.endswith('.txt'):
            loader = TextLoader(file_path)
        elif filename.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            continue # Skip unsupported file types

        docs.extend(loader.load())
        processed_files.append(filename)

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

    if api_key is None:
      try:
        api_key = secret_value
      except userdata.SecretNotFoundError:
        print("API key not found. Please set the GOOGLE_API_KEY in Colab secrets.")
        return None


    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
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
        print(f"Error creating QA chain: {e}")
        return None

def initialize_rag_system(folder_path, chunk_size=500, chunk_overlap=100):
    """Initialize the RAG system by loading documents, chunking, embedding, and storing"""
    documents, processed_files = load_documents_from_folder(folder_path)

    if not documents:
        return None, None

    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    vectorstore = embed_and_store(chunks)

    return vectorstore, processed_files

from diffusers import KandinskyV22Pipeline, KandinskyV22PriorPipeline
import torch
from PIL import Image
import os
import time

def load_kandinsky_pipeline():
    clear_gpu_memory()
    optimize_pytorch_memory
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

def generate_kandinsky_image(prompt, pipelines=None):
    """Generate image using Kandinsky 2.2"""
    try:
        if pipelines is None:
            prior_pipe, decoder_pipe = load_kandinsky_pipeline()
        else:
            prior_pipe, decoder_pipe = pipelines

        if prior_pipe is None or decoder_pipe is None:
            print("Pipelines not loaded properly")
            return None

        print(f"Generating image for prompt: {prompt}")
        negative_prompt = "low quality, bad quality, blurry, pixelated, distorted"

        # Generate embeddings using prior pipeline
        prior_output = prior_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=1.0,
            num_inference_steps=25
        )
        
        # Extract embeddings properly
        if hasattr(prior_output, 'image_embeds'):
            image_embeds = prior_output.image_embeds
            negative_image_embeds = prior_output.negative_image_embeds
        else:
            # Alternative extraction method
            image_embeds = prior_output[0] if isinstance(prior_output, (tuple, list)) else prior_output.image_embeds
            negative_image_embeds = prior_output[1] if isinstance(prior_output, (tuple, list)) and len(prior_output) > 1 else prior_output.negative_image_embeds

        # Generate image using decoder pipeline
        result = decoder_pipe(
            image_embeds=image_embeds,
            negative_image_embeds=negative_image_embeds,
            height=512,  # Reduced from 768 to avoid memory issues
            width=512,   # Reduced from 768 to avoid memory issues
            num_inference_steps=50,
            guidance_scale=4.0
        )
        
        # Extract the image properly
        if hasattr(result, 'images'):
            image = result.images[0]
        elif isinstance(result, (list, tuple)):
            image = result[0]
        else:
            image = result
            
        print("Image generated successfully")
        return image

    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
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

def initialize_tts():
    """Initialize Text-to-Speech engine"""
    global tts_engine
    print("AUDIO_AVAILABLE init tts")
    print(AUDIO_AVAILABLE)
    if AUDIO_AVAILABLE and tts_engine is None:
        try:
            tts_engine = pyttsx3.init()
            # Set properties
            voices = tts_engine.getProperty('voices')
            if voices:
                # Try to find a female voice for storytelling
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        tts_engine.setProperty('voice', voice.id)
                        break
            
            tts_engine.setProperty('rate', 180)  # Slower for storytelling
            tts_engine.setProperty('volume', 0.9)
            return True
        except Exception as e:
            print(f"Failed to initialize TTS: {e}")
            return False
    return AUDIO_AVAILABLE

def speech_to_text(audio_file):
    """Convert speech to text using SpeechRecognition"""
    print("AUDIO_AVAILABLE stt")
    print(AUDIO_AVAILABLE)
    if not AUDIO_AVAILABLE:
        return "Audio libraries not available"
    
    try:
        recognizer = sr.Recognizer()
        
        # Load audio file
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        # Convert to text
        text = recognizer.recognize_google(audio_data)
        return text
    
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition: {e}"
    except Exception as e:
        return f"Error processing audio: {e}"

def text_to_speech(text, output_path="response_audio.wav"):
    """Convert text to speech and save as audio file"""
    global tts_engine
    
    if not AUDIO_AVAILABLE or tts_engine is None:
        return None
    
    try:
        # Create temporary file for audio output
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()
        
        # Save audio to file
        tts_engine.save_to_file(text, temp_path)
        tts_engine.runAndWait()
        
        return temp_path
    
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None

def process_audio_conversation(audio_input, progress=gr.Progress()):
    """Process audio input and generate audio response with image"""
    global qa_chain, kandinsky_pipelines, tts_engine
    
    if not AUDIO_AVAILABLE:
        return "Audio processing not available", None, None, "Please install required audio libraries"
    
    if audio_input is None:
        return "No audio input provided", None, None, ""
    
    progress(0.1, desc="Converting speech to text...")
    
    # Convert audio to text
    try:
        question_text = speech_to_text(audio_input)
        if "Error" in question_text or "Could not" in question_text:
            return question_text, None, None, ""
    except Exception as e:
        return f"Audio processing error: {e}", None, None, ""
    
    progress(0.3, desc="Processing your question...")
    
    # Check if RAG system is ready
    if qa_chain is None:
        return "RAG system not initialized. Please setup RAG first.", None, None, question_text
    
    try:
        # Get response from RAG system
        response = qa_chain.invoke({"query": question_text})
        story_response = response['result']
        source_docs = [doc.metadata for doc in response['source_documents']]
        
        progress(0.5, desc="Extracting image prompt...")
        
        # Extract image prompt
        image_prompt_match = re.search(r"IMAGE_PROMPT:(.*)", story_response)
        generated_image = None
        
        if image_prompt_match:
            image_prompt = image_prompt_match.group(1).strip()
            
            progress(0.7, desc="Generating image...")
            
            # Load pipelines if needed
            if kandinsky_pipelines is None or kandinsky_pipelines[0] is None:
                kandinsky_pipelines = load_kandinsky_pipeline()
            
            if kandinsky_pipelines[0] is not None and kandinsky_pipelines[1] is not None:
                generated_image = generate_kandinsky_image(image_prompt, kandinsky_pipelines)
        
        progress(0.9, desc="Converting response to speech...")
        
        # Convert response to speech
        # Remove IMAGE_PROMPT from the spoken text
        spoken_text = re.sub(r"IMAGE_PROMPT:.*", "", story_response).strip()
        
        audio_response = text_to_speech(spoken_text)
        
        progress(1.0, desc="Complete!")
        
        source_info = f"Sources: {source_docs}" if source_docs else "No sources found"
        
        return story_response, generated_image, audio_response, question_text
        
    except Exception as e:
        print(f"Error in audio conversation: {e}")
        return f"Error processing request: {e}", None, None, question_text


def setup_rag_system(folder_path, chunk_size, chunk_overlap, progress=gr.Progress()):
    """Setup RAG system through Gradio"""
    global current_vectorstore, qa_chain
    
    progress(0.1, desc="Loading existing vectorstore...")
    
    # Try to load existing vectorstore first
    current_vectorstore = load_vectorstore()
    
    if current_vectorstore is None:
        progress(0.3, desc="No existing vectorstore found. Processing documents...")
        
        if not os.path.exists(folder_path):
            return f"Error: Folder '{folder_path}' does not exist!", ""
        
        progress(0.5, desc="Initializing RAG system...")
        current_vectorstore, processed_files = initialize_rag_system(folder_path, chunk_size, chunk_overlap)
        
        if current_vectorstore is None:
            return "Error: Failed to initialize RAG system. No documents found or processed.", ""
        
        status = f"Successfully processed {len(processed_files)} files: {', '.join(processed_files)}"
    else:
        status = "Loaded existing vectorstore successfully!"
    
    progress(0.8, desc="Creating QA chain...")
    qa_chain = create_qa_chain(current_vectorstore)
    
    if qa_chain is None:
        return "Error: Failed to create QA chain. Check your API key.", ""
    
    progress(1.0, desc="RAG system ready!")
    return status, "RAG system is ready! You can now ask questions."

def setup_kandinsky_pipeline(progress=gr.Progress()):
    """Setup Kandinsky pipeline through Gradio"""
    global kandinsky_pipelines
    
    progress(0.2, desc="Loading Kandinsky prior pipeline...")
    kandinsky_pipelines = load_kandinsky_pipeline()
    
    if kandinsky_pipelines[0] is None or kandinsky_pipelines[1] is None:
        return "Error: Failed to load Kandinsky pipelines!"
    
    progress(1.0, desc="Kandinsky pipelines ready!")
    return "Kandinsky pipelines loaded successfully!"

def ask_question_and_generate(question, progress=gr.Progress()):
    """Ask question and generate image through Gradio"""
    global qa_chain, kandinsky_pipelines
    
    if qa_chain is None:
        return "Error: RAG system not initialized. Please setup RAG first.", None, ""
    
    progress(0.2, desc="Processing question...")
    
    try:
        response = qa_chain.invoke({"query": question})
        story_response = response['result']
        source_docs = [doc.metadata for doc in response['source_documents']]
        
        progress(0.5, desc="Extracting image prompt...")
        
        # Extract image prompt
        image_prompt_match = re.search(r"IMAGE_PROMPT:(.*)", story_response)
        
        if image_prompt_match:
            image_prompt = image_prompt_match.group(1).strip()
            print(f"Extracted image prompt: {image_prompt}")
            
            progress(0.7, desc="Generating image...")
            
            # Load pipelines if not already loaded
            if kandinsky_pipelines is None or kandinsky_pipelines[0] is None:
                print("Loading Kandinsky pipelines...")
                kandinsky_pipelines = load_kandinsky_pipeline()
            
            if kandinsky_pipelines[0] is not None and kandinsky_pipelines[1] is not None:
                generated_image = generate_kandinsky_image(image_prompt, kandinsky_pipelines)
                progress(1.0, desc="Complete!")
                
                source_info = f"Sources: {source_docs}" if source_docs else "No sources found"
                
                if generated_image is not None:
                    return story_response, generated_image, source_info
                else:
                    return story_response, None, f"Story generated but image generation failed. Sources: {source_docs}"
            else:
                return story_response, None, f"Story generated but failed to load image pipelines. Sources: {source_docs}"
        else:
            return story_response, None, f"Story generated but no image prompt found. Sources: {source_docs}"
            
    except Exception as e:
        print(f"Error in ask_question_and_generate: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", None, ""

def generate_image_only(prompt, progress=gr.Progress()):
    """Generate image from custom prompt"""
    global kandinsky_pipelines
    
    if not prompt.strip():
        return None
    
    progress(0.3, desc="Loading pipelines if needed...")
    
    # Load pipelines if not already loaded
    if kandinsky_pipelines is None or kandinsky_pipelines[0] is None:
        print("Loading Kandinsky pipelines for custom image generation...")
        kandinsky_pipelines = load_kandinsky_pipeline()
    
    if kandinsky_pipelines[0] is None or kandinsky_pipelines[1] is None:
        print("Failed to load Kandinsky pipelines")
        return None
    
    progress(0.7, desc="Generating image...")
    
    generated_image = generate_kandinsky_image(prompt, kandinsky_pipelines)
    
    progress(1.0, desc="Complete!")
    return generated_image

# Create Gradio Interface
with gr.Blocks(title="RAG + Kandinsky Storyteller", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📚 RAG + Kandinsky Storyteller")
    gr.Markdown("Generate funny stories from your documents and create images to go with them!")
    
    with gr.Tab("Setup"):
        gr.Markdown("## 🛠️ System Setup")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### RAG System Setup")
                folder_input = gr.Textbox(
                    label="Documents Folder Path", 
                    value="story_books",
                    placeholder="Enter path to your documents folder"
                )
                chunk_size = gr.Slider(
                    minimum=100, maximum=1000, value=500, step=50,
                    label="Chunk Size"
                )
                chunk_overlap = gr.Slider(
                    minimum=0, maximum=200, value=100, step=25,
                    label="Chunk Overlap"
                )
                setup_rag_btn = gr.Button("Setup RAG System", variant="primary")
                rag_status = gr.Textbox(label="RAG Status", interactive=False)
                rag_ready = gr.Textbox(label="System Ready", interactive=False)
            
            with gr.Column():
                gr.Markdown("### Kandinsky Image Generation Setup")
                setup_kandinsky_btn = gr.Button("Load Kandinsky Pipelines", variant="primary")
                kandinsky_status = gr.Textbox(label="Kandinsky Status", interactive=False)
    
    with gr.Tab("Story Generation"):
        gr.Markdown("## 📖 Ask Questions & Generate Stories")
        
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Ask about characters, plots, or scenes from your documents...",
            lines=2
        )
        
        ask_btn = gr.Button("Generate Story & Image", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                story_output = gr.Textbox(
                    label="Generated Story",
                    lines=10,
                    max_lines=15
                )
                source_output = gr.Textbox(
                    label="Source Information",
                    lines=3
                )
            
            with gr.Column():
                image_output = gr.Image(
                    label="Generated Image",
                    height=400
                )
    
    with gr.Tab("🎤 Audio Conversation"):
        gr.Markdown("## 🎙️ Voice-to-Voice Storytelling")
        gr.Markdown("Record your question, get an AI-generated story with image, and hear the response!")
        
        if not AUDIO_AVAILABLE:
            gr.Markdown("⚠️ **Audio libraries not available.** Install required packages:")
            # gr.Code("pip install SpeechRecognition pyttsx3 librosa soundfile", language="bash")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="🎤 Record Your Question",
                    type="filepath",
                    sources=["microphone"]
                )
                
                process_audio_btn = gr.Button(
                    "🎬 Generate Story & Response", 
                    variant="primary", 
                    size="lg"
                )
                
                recognized_text = gr.Textbox(
                    label="📝 What I heard you say:",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column():
                audio_story_output = gr.Textbox(
                    label="📖 Generated Story",
                    lines=8,
                    max_lines=12
                )
                
                audio_response = gr.Audio(
                    label="🔊 AI Voice Response",
                    type="filepath"
                )
        
        with gr.Row():
            audio_image_output = gr.Image(
                label="🎨 Generated Story Image",
                height=400
            )
        
        # Audio conversation event
        process_audio_btn.click(
            fn=process_audio_conversation,
            inputs=audio_input,
            outputs=[audio_story_output, audio_image_output, audio_response, recognized_text]
        )
        
        gr.Markdown("---")
        
        # Detailed information about improvements
        with gr.Accordion("🚀 **Future Enhancements: Real-time Streaming & Live Assistant**", open=False):
            gr.Markdown("""
            ## 🌟 **Current Limitations & How to Improve**
            
            ### **Current System:**
            - **Non-streaming**: Record → Process → Respond (batch processing)
            - **High latency**: 10-30 seconds for complete response
            - **Static conversation**: One question, one answer
            
            ---
            
            ## 🎯 **Next-Level Improvements**
            
            ### **1. Real-time Speech Streaming** 🎙️
            
            **Technologies to implement:**
            ```python
            # WebRTC for real-time audio streaming
            import webrtc_streamer
            
            # Real-time speech recognition
            from google.cloud import speech
            from azure.cognitiveservices.speech import SpeechRecognizer
            
            # Streaming ASR
            def streaming_speech_recognition():
                # Continuous listening with voice activity detection
                # Real-time transcription as you speak
                pass
            ```
            
            **Benefits:**
            - ✅ **Instant transcription** as you speak
            - ✅ **Voice activity detection** (auto start/stop)
            - ✅ **No manual recording** needed
            
            ### **2. Streaming LLM Response** ⚡
            
            **Implementation approach:**
            ```python
            # Streaming response generation
            def streaming_llm_response():
                for chunk in llm.stream(prompt):
                    yield chunk  # Real-time token generation
                    
            # Concurrent TTS generation
            async def stream_tts():
                # Generate audio as text streams in
                # Play audio chunks in real-time
                pass
            ```
            
            **Benefits:**
            - ✅ **Immediate response start** (no waiting for complete generation)
            - ✅ **Natural conversation flow**
            - ✅ **Reduced perceived latency**
            
            ### **3. Advanced Voice Assistant Features** 🤖
            
            **Wake word detection:**
            ```python
            # Always-listening mode
            from pocketsphinx import LiveSpeech
            
            def wake_word_detection():
                # "Hey Story Bot, tell me about..."
                # Automatic conversation initiation
                pass
            ```
            
            **Conversation memory:**
            ```python
            # Context-aware conversations
            conversation_history = []
            
            def contextual_response(new_question):
                # Remember previous questions
                # Build upon previous stories
                # Create connected narratives
                pass
            ```
            
            ### **4. Multi-modal Streaming** 🎭
            
            **Real-time image generation:**
            ```python
            # Streaming image generation
            def stream_image_generation():
                # Generate images as story unfolds
                # Progressive image refinement
                # Multiple scene generations
                pass
            ```
            
            **Synchronized multimedia:**
            - 🎵 **Background music** generation
            - 🎬 **Story scene transitions**
            - 📱 **Mobile app** with better UX
            
            ---
            
            ## 🛠️ **Technical Implementation Roadmap**
            
            ### **Phase 1: Basic Streaming (2-3 weeks)**
            ```bash
            pip install webrtc-vad speechrecognition-streaming
            pip install azure-cognitiveservices-speech
            pip install google-cloud-speech
            ```
            
            ### **Phase 2: Advanced Features (1-2 months)**
            - **WebSocket connections** for real-time data
            - **FastAPI backend** with streaming endpoints
            - **React frontend** with real-time audio handling
            - **Redis** for conversation state management
            
            ### **Phase 3: Production Ready (2-3 months)**
            - **Kubernetes deployment** with auto-scaling
            - **CDN integration** for global low-latency
            - **Mobile app** (React Native/Flutter)
            - **Voice cloning** for personalized narrators
            
            ---
            
            ## 🎯 **Recommended Architecture**
            
            ```
            User Voice Input
                ↓
            WebRTC Streaming
                ↓
            Real-time ASR (Google/Azure)
                ↓
            Streaming LLM (GPT-4/Gemini)
                ↓
            ┌─ Concurrent TTS Generation
            └─ Parallel Image Generation
                ↓
            WebSocket Response Stream
                ↓
            Real-time Audio/Visual Output
            ```
            
            ### **Key Technologies:**
            - **Frontend**: React + WebRTC + Web Audio API
            - **Backend**: FastAPI + WebSockets + Redis
            - **AI Services**: OpenAI GPT-4 Turbo + Azure Speech
            - **Image Gen**: DALL-E 3 or Midjourney API
            - **Infrastructure**: AWS/GCP with edge locations
            
            ---
            
            ## 💡 **Cool Features to Add**
            
            ### **Interactive Storytelling:**
            - 🎭 **"What happens next?"** - Choose your adventure
            - 🎪 **Character voices** - Different voices for each character
            - 🎨 **Story illustrations** - Multiple images per story
            - 🎵 **Dynamic music** - AI-generated background scores
            
            ### **Personalization:**
            - 👤 **Voice profiles** - Remember user preferences
            - 📚 **Story library** - Save and replay favorite stories
            - 🎯 **Age-appropriate** content filtering
            - 🌍 **Multi-language** support
            
            ### **Social Features:**
            - 👨‍👩‍👧‍👦 **Family sharing** - Stories for multiple users
            - 🏆 **Story competitions** - Community-generated content
            - 📱 **Mobile notifications** - Daily story suggestions
            
            ---
            
            ## 🚦 **Getting Started with Improvements**
            
            ### **Week 1: Basic Streaming**
            1. Implement WebRTC audio streaming
            2. Add real-time speech recognition
            3. Test latency improvements
            
            ### **Week 2-3: Response Streaming**
            1. Implement LLM response streaming
            2. Add concurrent TTS generation
            3. Optimize audio playback
            
            ### **Month 2: Advanced Features**
            1. Add conversation memory
            2. Implement wake word detection
            3. Build mobile-responsive interface
            
            **Ready to build the future of AI storytelling? 🚀**
            """)
    
    with gr.Tab("Custom Image Generation"):
        gr.Markdown("## 🎨 Generate Custom Images")
        
        custom_prompt = gr.Textbox(
            label="Image Prompt",
            placeholder="Describe the image you want to generate...",
            lines=3
        )
        
        generate_img_btn = gr.Button("Generate Image", variant="primary")
        custom_image_output = gr.Image(label="Generated Image", height=400)
    
    with gr.Tab("Examples"):
        gr.Markdown("## 💡 Example Questions")
        gr.Markdown("""
        Try asking questions like:
        - "Tell me about Aladdin and his adventures"
        - "Who is Alice and what happens to her?"
        - "Describe the main character in [story name]"
        - "What magical elements appear in the stories?"
        - "Tell me about the villains in these tales"
        """)
        
        example_questions = [
            "The Story of Aladdin and the Magic Lamp?",
            "Who is Alice and what adventures does she have?",
            "Tell me about Cinderella's story",
            "What happens in Little Red Riding Hood?",
            "Describe the Three Little Pigs story"
        ]
        
        for question in example_questions:
            gr.Button(question, variant="secondary").click(
                fn=lambda q=question: q,
                outputs=question_input
            )
    
    # Event handlers
    setup_rag_btn.click(
        fn=setup_rag_system,
        inputs=[folder_input, chunk_size, chunk_overlap],
        outputs=[rag_status, rag_ready]
    )
    
    setup_kandinsky_btn.click(
        fn=setup_kandinsky_pipeline,
        outputs=kandinsky_status
    )
    
    ask_btn.click(
        fn=ask_question_and_generate,
        inputs=question_input,
        outputs=[story_output, image_output, source_output]
    )
    
    generate_img_btn.click(
        fn=generate_image_only,
        inputs=custom_prompt,
        outputs=custom_image_output
    )

if __name__ == "__main__":
    # Initialize TTS engine on startup
    if initialize_tts():
        print("TTS engine initialized successfully")
    else:
        print("TTS engine initialization failed")
    
    # Launch the app
    app.launch(
        share=True,  # Creates a public link
        debug=True,
        server_name="0.0.0.0",  # Makes it accessible from other devices
        server_port=7860
    )
