# DocuChat – PDF Summarizer & Q&A Chatbot
DocuChat is a Streamlit-powered web app that lets you:

📂 Upload a PDF document
📋 Get an automatic summary
💬 Ask questions about its content (chatbot)
🧠 Generate a mindmap of key concepts
🔊 Listen to answers with text-to-speech
🎤 Ask questions with voice input

# Features⚡

✅ Upload and process PDF documents
✅ Automatic text summarization
✅ AI-powered question answering (Google Gemini API)
✅ Mindmap generation using NetworkX & Matplotlib
✅ Wordclouds for keyword insights
✅ Voice input (speech-to-text) and audio answers (text-to-speech)
✅ Downloadable summary report (PDF)

# Tech Stack🛠️
Streamlit – Web app framework
Google Gemini API – AI for summarization & Q&A
PyPDF2 – PDF text extraction
SpeechRecognition – Voice input
gTTS – Text-to-speech
Matplotlib / NetworkX – Mindmap visualization
Seaborn, Plotly – Charts & insights
WordCloud – Word clouds
ReportLab – PDF report generation

# Installation📂 
Clone the repository
git clone https:/simran487/github.com//DocuChat.git
cd DocuChat

# Create a virtual environment
python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
Create a .env file in the project root:
GEMINI_API_KEY=your_google_gemini_api_key_here

# ▶️ Run the App
streamlit run app.py

# Open the app in your browser:
👉 http://localhost:8501

# 📸 Screenshots
<img width="1901" height="879" alt="image" src="https://github.com/user-attachments/assets/f26ecf87-fca8-49e7-a474-6835c13f6abc" />
<img width="1898" height="855" alt="image" src="https://github.com/user-attachments/assets/a46061e4-7f09-4bbf-a306-42d9ae61d387" />



✨ With DocuChat, reading long PDFs becomes easy, interactive, and fun!
