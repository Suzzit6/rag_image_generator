# RAG Image Generator with Voice Interaction

This project combines Retrieval Augmented Generation (RAG) with image generation capabilities and voice interaction to create an engaging storytelling experience. It uses various documents as sources of knowledge, generates humorous stories based on user queries, and creates accompanying images using Kandinsky AI.

## 🌟 Features

- **Document Processing**: Load and process PDF, TXT, and DOCX documents
- **RAG System**: Retrieval-based generation using vector embeddings
- **Funny Storytelling**: AI-generated humorous stories from document contents
- **Image Generation**: Create images that match the story using Kandinsky 2.2
- **Voice Interaction**: Ask questions using voice and hear spoken responses
- **Multiple Interfaces**: Streamlit and Gradio interfaces for user interaction

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for image generation)
- Internet connection for API access

## 🔧 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Suzzit6/rag_image_generator
   cd rag_image_generator
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   - Create a .env file in the root directory
   - Add your Google Gemini API key:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## 🔧 Kaggle Setup

1. **Add Google Gemini API key to Kaggle secrets**:
   - Go to your Kaggle account settings
   - Navigate to "API" section
   - Under "Secrets", add a new secret with label `GEMINI_API_KEY` and enter your API key

2. **Import the story_books data**:
   - In your Kaggle notebook, click on the "Data" tab
   - Click "+ Add Data" 
   - Select "Import from GitHub"
   - Enter the URL: `https://github.com/Suzzit6/rag_image_generator`
   - This will import the necessary story books for the RAG system

3. **Run the notebook**:
   - The code will automatically retrieve the API key from Kaggle secrets
   - Adjust the folder path to match where Kaggle placed the imported data

## 📚 Project Structure

```
.
├── .env                  # Environment variables and API keys
├── .gitignore            # Git ignore file
├── gradio_.py            # Gradio web interface
├── main.py               # Core functionality implementation
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
├── streamlit.py          # Streamlit web interface
├── rag_vectorstore/      # Vector database storage
│   ├── index.faiss       # FAISS index
│   └── index.pkl         # Pickle file for metadata
└── story_books/          # Sample documents
    ├── alice_in_wonderland.pdf
    ├── Gullivers_Travels.pdf
    └── The_Arabian_Nights.pdf
```

## 🚀 Usage

### Using Streamlit Interface

1. **Launch the Streamlit app**:
   ```bash
   python -m streamlit run streamlit.py
   ```

2. **Initialize the RAG system**:
   - Click "Initialize/Reload RAG System" in the sidebar
   - Wait for the vector store to be created or loaded

3. **Ask questions**:
   - Type your question in the text input box
   - Click "Generate Story & Image" to get a response
   - The system will generate a humorous story and a matching image

### Using Gradio Interface

1. **Launch the Gradio app**:
   ```bash
   python gradio_.py
   ```

2. **Set up the system**:
   - Go to the "Setup" tab
   - Configure the document folder path (default: "story_books")
   - Click "Setup RAG System" and "Load Kandinsky Pipelines"

3. **Generate stories and images**:
   - Go to the "Story Generation" tab
   - Enter your question and click "Generate Story & Image"
   - View the response and generated image

4. **Voice interaction** (if audio libraries are installed):
   - Go to the "Audio Conversation" tab
   - Record your question using the microphone
   - Click "Generate Story & Response"
   - Listen to the spoken response and view the generated image

## 🔈 Voice Interaction Setup

For voice interaction capabilities, install additional packages:

```bash
pip install SpeechRecognition pyttsx3 librosa soundfile
```

## 🧠 How It Works

### RAG System

1. **Document Loading**: Reads documents from the specified folder
2. **Chunking**: Splits documents into manageable chunks
3. **Embedding**: Creates vector embeddings using HuggingFace's all-MiniLM-L6-v2
4. **Storage**: Stores vectors in a FAISS database for efficient retrieval
5. **Retrieval**: Finds relevant document chunks based on user queries
6. **Generation**: Uses Google Gemini 2.0 Flash to create responses with relevant context

### Image Generation

1. **Prompt Extraction**: Extracts image prompt from the generated story
2. **Model Loading**: Uses Kandinsky 2.2 (prior and decoder pipelines)
3. **Image Creation**: Generates an image based on the extracted prompt
4. **Display**: Shows the generated image alongside the story

### Voice Processing

1. **Speech Recognition**: Converts spoken input to text
2. **Text-to-Speech**: Converts generated stories to spoken audio
3. **Interactive Interface**: Enables voice-based conversations

## 🛣️ Roadmap

- Streaming voice conversations with real-time responses
- More diverse image generation options
- Support for additional document formats
- Enhanced memory for contextual conversations
- Mobile application development

## 🔧 Troubleshooting

- **Memory Issues**: Reduce image resolution if encountering CUDA out-of-memory errors
- **API Limits**: Be aware of Google API usage limits for Gemini
- **Voice Recognition**: Ensure you have a good microphone for better recognition
- **Image Generation**: Requires significant GPU memory; reduce dimensions if needed

## 📜 License

[Specify your license here]

## 👏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG implementation
- [Kandinsky](https://github.com/ai-forever/Kandinsky-2) for the image generation model
- [Hugging Face](https://huggingface.co/) for the embedding models
- [Google Gemini](https://ai.google.dev/) for the LLM API

---

Made with ❤️ using LangChain, Streamlit, Gradio, and Kandinsky