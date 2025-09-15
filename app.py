import streamlit as st
import PyPDF2
import os
import io
import tempfile
import re
from collections import Counter
from gtts import gTTS
import speech_recognition as sr
from audio_recorder_streamlit import audio_recorder
from datetime import datetime
import time
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib import colors
from PIL import Image
import pandas as pd
import json
from dotenv import load_dotenv



# load variables from .env
load_dotenv()


try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.error("Gemini AI not available. Please install google-genai package.")

# Initialize Gemini AI
if GEMINI_AVAILABLE:
    try:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    except Exception as e:
        st.error(f"Failed to initialize Gemini client: {str(e)}")
        client = None
else:
    client = None

# -------------------------------
# Helper Functions
# -------------------------------

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def get_most_frequent_words(text, num_words=20):
    """Get most frequent meaningful words from text"""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 
        'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
        'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
        'theirs', 'themselves'
    }
    
    # Clean text and get words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [word for word in words if word not in stop_words]
    
    # Get most frequent words
    word_counts = Counter(words)
    return word_counts.most_common(num_words)

def extract_key_sentences(text, num_sentences=5):
    """Extract key sentences based on word frequency and position"""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= num_sentences:
        return sentences
    
    # Get frequent words
    frequent_words = dict(get_most_frequent_words(text, 30))
    
    # Score sentences based on frequent word occurrence
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        score = 0
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        
        # Score based on frequent words
        for word in words:
            if word in frequent_words:
                score += frequent_words[word]
        
        # Boost score for early sentences (introduction often important)
        if i < 3:
            score *= 1.5
        
        # Normalize by sentence length
        if len(words) > 0:
            score = score / len(words)
        
        sentence_scores.append((sentence, score))
    
    # Sort by score and return top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    return [sent[0] for sent in sentence_scores[:num_sentences]]

def summarize_text(text):
    """Create a comprehensive AI-powered summary using Gemini"""
    if not text or len(text.strip()) < 100:
        return "Document too short to summarize effectively."
    
    # Check if Gemini is available and working
    if not GEMINI_AVAILABLE or client is None:
        return fallback_summarize_text(text)
    
    try:
        # Create a comprehensive prompt for AI summarization
        prompt = f"""
        Please provide a comprehensive and well-structured summary of the following document. 
        
        Requirements:
        1. Start with a clear overview of what the document is about
        2. Identify and organize the main topics and key points
        3. Present information in a logical, coherent manner
        4. Include important details, concepts, and conclusions
        5. Use clear headings and bullet points for better readability
        6. Ensure the summary flows naturally from beginning to end
        
        Document content to summarize:
        {text[:20000]}  # Increased limit for better context
        
        Please create a meaningful, well-organized summary that captures the essence and key information of this document.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text.strip()
        else:
            # Fallback to basic summarization if AI fails
            return fallback_summarize_text(text)

    except Exception as e:
        st.error(f"AI summarization error: {str(e)}")
        # Fallback to basic summarization
        return fallback_summarize_text(text)

def fallback_summarize_text(text):
    """Fallback summary method using key sentence extraction"""
    # Get key sentences
    key_sentences = extract_key_sentences(text, 6)
    
    # Get most frequent words for context
    frequent_words = get_most_frequent_words(text, 10)
    key_terms = [word for word, count in frequent_words[:8]]
    
    # Create summary
    summary_parts = []
    
    # Add overview
    summary_parts.append("**Document Overview:**")
    summary_parts.append("This document covers the following key areas and concepts.")
    summary_parts.append("")
    
    # Add key terms context
    if key_terms:
        summary_parts.append(f"**Main Topics:** {', '.join(key_terms)}")
        summary_parts.append("")
    
    # Add key points
    if key_sentences:
        summary_parts.append("**Key Points:**")
        for i, sentence in enumerate(key_sentences, 1):
            summary_parts.append(f"{i}. {sentence.strip()}")
    
    return "\n".join(summary_parts) if summary_parts else "Unable to generate summary."

def answer_question_with_ai(context, question):
    """Answer question using Gemini AI with context from PDF"""
    # Check if Gemini is available and working
    if not GEMINI_AVAILABLE or client is None:
        return simple_qa_search(context, question)
    
    try:
        # Create a comprehensive prompt for Gemini
        prompt = f"""
        You are an intelligent document assistant. Based on the following document content, please answer the user's question accurately and helpfully.

        Document Content:
        {context[:15000]}  # Increased context limit for better answers

        User Question: {question}

        Please provide a clear, accurate answer based on the information in the document. If the answer is not directly available in the document, say so clearly. Keep your response concise but informative.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text.strip()
        else:
            return "I couldn't generate a response. Please try rephrasing your question."

    except Exception as e:
        st.error(f"AI Error: {str(e)}")
        # Fallback to simple keyword matching if AI fails
        return simple_qa_search(context, question)

def simple_qa_search(context, question):
    """Simple question answering using keyword matching and context extraction"""
    context_lower = context.lower()
    question_lower = question.lower()
    
    # Extract question keywords (excluding common question words)
    question_words = re.findall(r'\b[a-zA-Z]{3,}\b', question_lower)
    stop_words = {'what', 'when', 'where', 'who', 'why', 'how', 'does', 'did', 'can', 'will', 'would'}
    keywords = [word for word in question_words if word not in stop_words]
    
    if not keywords:
        return "Please ask a more specific question with clear keywords."
    
    # Find sentences containing question keywords
    sentences = re.split(r'[.!?]+', context)
    relevant_sentences = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matches = sum(1 for keyword in keywords if keyword in sentence_lower)
        
        if matches > 0:
            relevant_sentences.append((sentence.strip(), matches))
    
    if not relevant_sentences:
        return f"I couldn't find information about '{' '.join(keywords)}' in the document."
    
    # Sort by relevance and return best matches
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 most relevant sentences
    answer_parts = []
    for sentence, _ in relevant_sentences[:3]:
        if sentence and len(sentence) > 10:
            answer_parts.append(sentence)
    
    if answer_parts:
        return " ".join(answer_parts)
    else:
        return "I found related information but couldn't extract a clear answer."

def answer_question(context, question):
    """Answer question based on context using simple keyword matching"""
    return simple_qa_search(context, question)

def create_word_frequency_chart(text):
    """Create a word frequency bar chart"""
    frequent_words = get_most_frequent_words(text, 15)
    if not frequent_words:
        return None
    
    words = [item[0] for item in frequent_words]
    counts = [item[1] for item in frequent_words]
    
    fig = go.Figure(data=[
        go.Bar(x=words, y=counts, 
               marker_color='lightblue',
               text=counts,
               textposition='auto',)
    ])
    
    fig.update_layout(
        title="Most Frequent Words in Document",
        xaxis_title="Words",
        yaxis_title="Frequency",
        height=400,
        showlegend=False
    )
    
    return fig

def create_word_cloud(text):
    """Create a word cloud visualization"""
    try:
        # Clean text for word cloud
        clean_text = ' '.join([word for word in re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())])
        
        if len(clean_text) < 10:
            return None
            
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            relative_scaling='auto',
            random_state=42
        ).generate(clean_text)
        
        # Save to bytes
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        return img_buffer
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def create_mindmap_with_ai(text):
    """Create an AI-powered mindmap of document concepts"""
    # Check if Gemini is available and working
    if not GEMINI_AVAILABLE or client is None:
        return create_simple_mindmap(text)
    
    try:
        # Create prompt for AI to extract key concepts and relationships
        prompt = f"""
        Analyze the following document and create a structured mindmap representation.
        
        Please extract:
        1. Main topic/theme of the document
        2. 5-8 key subtopics or categories 
        3. 2-3 important details for each subtopic
        4. How these concepts relate to each other
        
        Format your response as a JSON structure like this:
        {{
            "main_topic": "Document Main Theme",
            "subtopics": [
                {{
                    "name": "Subtopic 1",
                    "details": ["Detail 1", "Detail 2", "Detail 3"],
                    "connections": ["Subtopic 2", "Subtopic 3"]
                }},
                ...
            ]
        }}
        
        Document content:
        {text[:15000]}
        
        Please provide only the JSON response, no other text.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        
        if response.text:
            try:
                # Extract JSON from response using robust parsing
                response_text = response.text.strip()
                
                # Remove markdown code blocks if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]
                
                # Find first valid JSON object using regex
                import re as regex_module
                json_match = regex_module.search(r'\{.*\}', response_text, regex_module.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    mindmap_data = json.loads(json_str)
                    
                    # Validate required keys
                    if "main_topic" in mindmap_data and "subtopics" in mindmap_data:
                        return create_mindmap_visualization(mindmap_data)
                    else:
                        return create_simple_mindmap(text)
                else:
                    return create_simple_mindmap(text)
                    
            except (json.JSONDecodeError, KeyError, AttributeError):
                # Silently fall back to simple mindmap without user-facing errors
                return create_simple_mindmap(text)
        else:
            return create_simple_mindmap(text)
            
    except Exception as e:
        st.error(f"AI mindmap error: {str(e)}")
        return create_simple_mindmap(text)

def create_simple_mindmap(text):
    """Create a simple mindmap using keyword extraction"""
    # Get key topics
    frequent_words = get_most_frequent_words(text, 12)
    topics = [word for word, count in frequent_words[:8]]
    
    # Create simple structure
    mindmap_data = {
        "main_topic": "Document Overview",
        "subtopics": [
            {
                "name": topic.title(),
                "details": [f"Related to {topic}"],
                "connections": [topics[(i+1) % len(topics)].title() for i in range(min(2, len(topics)-1))]
            }
            for i, topic in enumerate(topics)
        ]
    }
    
    return create_mindmap_visualization(mindmap_data)

def create_mindmap_visualization(mindmap_data):
    """Create interactive mindmap visualization using networkx and plotly"""
    try:
        # Create network graph
        G = nx.Graph()
        
        # Add main topic as central node
        main_topic = mindmap_data.get("main_topic", "Document")
        G.add_node(main_topic, node_type="main", size=30)
        
        # Add subtopics and their details
        subtopics = mindmap_data.get("subtopics", [])
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        for i, subtopic in enumerate(subtopics):
            subtopic_name = subtopic.get("name", f"Topic {i+1}")
            color = colors[i % len(colors)]
            
            # Add subtopic node
            G.add_node(subtopic_name, node_type="subtopic", size=20, color=color)
            G.add_edge(main_topic, subtopic_name)
            
            # Add detail nodes
            details = subtopic.get("details", [])
            for detail in details[:3]:  # Limit to 3 details per subtopic
                if detail and len(detail.strip()) > 0:
                    detail_short = detail[:30] + "..." if len(detail) > 30 else detail
                    G.add_node(detail_short, node_type="detail", size=10, color=color)
                    G.add_edge(subtopic_name, detail_short)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare data for plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edges trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='rgba(100,100,100,0.5)'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes trace
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            node_data = G.nodes[node]
            if node_data.get('node_type') == 'main':
                node_colors.append('#2E86AB')
                node_sizes.append(40)
            elif node_data.get('node_type') == 'subtopic':
                node_colors.append(node_data.get('color', '#FF6B6B'))
                node_sizes.append(25)
            else:
                node_colors.append(node_data.get('color', '#A8DADC'))
                node_sizes.append(15)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            textfont=dict(size=10, color='#111111')
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Document Mindmap - Key Concepts & Relationships',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive mindmap showing document structure and relationships",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="#888", size=12)
                               ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white',
                           paper_bgcolor='white',
                           height=600))
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating mindmap visualization: {str(e)}")
        return None


def create_summary_pdf(text, summary, stats=None):
    """Create a downloadable PDF summary"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.darkgreen,
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("PDF Document Summary Report", title_style))
        story.append(Spacer(1, 20))
        
        
        # Summary Section
        story.append(Paragraph("Executive Summary", heading_style))
        
        # Process summary text for PDF
        summary_lines = summary.split('\n')
        for line in summary_lines:
            if line.strip():
                if line.startswith('**') and line.endswith('**'):
                    # Convert markdown headers to PDF headers
                    clean_line = line.replace('**', '')
                    story.append(Paragraph(clean_line, heading_style))
                elif line.startswith('•'):
                    # Convert bullet points
                    story.append(Paragraph(line, styles['Normal']))
                else:
                    story.append(Paragraph(line, styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Key Insights section
        frequent_words = get_most_frequent_words(text, 10)
        if frequent_words:
            story.append(Paragraph("Key Terms & Concepts", heading_style))
            terms_text = ", ".join([word for word, count in frequent_words])
            story.append(Paragraph(f"<i>{terms_text}</i>", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def speech_to_text(audio_bytes):
    """Convert speech to text using speech recognition"""
    try:
        recognizer = sr.Recognizer()
        
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file.flush()
            
            # Use speech recognition
            with sr.AudioFile(tmp_file.name) as source:
                audio = recognizer.record(source)
                try:
                    # Use Google Web Speech API if available
                    if hasattr(recognizer, 'recognize_google'):
                        text = recognizer.recognize_google(audio)
                        return text
                    else:
                        st.warning("Speech recognition not available. Please use text input.")
                        return None
                except sr.UnknownValueError:
                    st.warning("Could not understand the audio. Please speak clearly and try again.")
                    return None
                except sr.RequestError:
                    st.warning("Speech recognition service is unavailable. Please use text input instead.")
                    return None
                except AttributeError:
                    st.warning("Speech recognition method not available. Please use text input.")
                    return None
    except Exception as e:
        st.warning(f"Audio processing not available. Please use text input: {str(e)}")
        return None

def create_download_link(file_path, link_text):
    """Create download link for audio file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="response.mp3">{link_text}</a>'
    return href

# -------------------------------
# Streamlit App
# -------------------------------

def render_floating_chatbot():
    """Create a modern floating chatbot interface using Streamlit sidebar"""
    
    # Add CSS for modern chatbot styling
    st.markdown("""
    <style>
    .floating-chat {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        z-index: 1000;
        transition: all 0.3s ease;
    }
    
    .floating-chat:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 25px rgba(0,0,0,0.4);
    }
    
    .chat-input-modern {
        border: 2px solid #e0e0e0 !important;
        border-radius: 25px !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .chat-input-modern:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .modern-chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Floating chat icon (visual only - actual chat is in sidebar)
    st.markdown("""
    <div class="floating-chat" title="Chat with PDF Assistant">
        💬
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="PDF Summarizer & Q&A Chatbot",
        page_icon="📄",
        layout="wide"
    )
    
    # Render the floating chatbot
    render_floating_chatbot()
    
    st.title("📄 PDF Summarizer & Q&A Chatbot")
    st.markdown("Upload a PDF document to get an automatic summary and ask questions about its content!")
    
    # Initialize session state
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "pdf_summary" not in st.session_state:
        st.session_state.pdf_summary = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_uploaded" not in st.session_state:
        st.session_state.pdf_uploaded = False
    if "chatbot_question" not in st.session_state:
        st.session_state.chatbot_question = ""
    
    # Sidebar for PDF upload and Quick Chat
    with st.sidebar:
        st.header("📁 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
        
        # ChatGPT-like Interface
        st.markdown("---")
        st.markdown('<div class="chat-header">🤖 AI Chat Assistant</div>', unsafe_allow_html=True)
        
        if st.session_state.pdf_uploaded:
            st.success("✨ AI is ready to answer your questions!")
            
            # Initialize chat session state
            if "current_question" not in st.session_state:
                st.session_state.current_question = ""
            
            # Voice input section (records to text area)
            st.markdown("#### 🎤 Voice Input")
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#667eea",
                icon_name="microphone",
                icon_size="1x",
                key="voice_recorder"
            )
            
            if audio_bytes:
                with st.spinner("🎧 Converting speech to text..."):
                    voice_text = speech_to_text(audio_bytes)
                    if voice_text:
                        st.session_state.current_question = voice_text
                        st.rerun()
            
            # Text area for questions (like ChatGPT)
            st.markdown("#### 💬 Type or ask your question:")
            question = st.text_area(
                "Your question:",
                value=st.session_state.current_question,
                placeholder="Ask me anything about your document...\n\nFor example:\n- What is this document about?\n- Summarize the main points\n- What are the key findings?",
                height=120,
                key="question_input"
            )
            
            # Search button (like ChatGPT send button)
            col1, col2 = st.columns([3, 1])
            with col2:
                search_button = st.button("🔍 Search", use_container_width=True, type="primary")
            
            with col1:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.current_question = ""
                    st.rerun()
            
            # Process question when search button is clicked
            if search_button and question.strip():
                with st.spinner("🤖 AI is thinking..."):
                    # Use AI-powered answer function
                    answer = answer_question_with_ai(st.session_state.pdf_text, question.strip())
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question.strip(),
                        "answer": answer,
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                    
                    # Generate audio response
                    try:
                        audio_file = text_to_speech(answer)
                        if audio_file:
                            st.session_state.chat_history[-1]["audio_file"] = audio_file
                    except:
                        pass  # Continue without audio if TTS fails
                    
                    # Clear the question for next input
                    st.session_state.current_question = ""
                    st.rerun()
            
            # Display chat history (ChatGPT style)
            if st.session_state.chat_history:
                st.markdown("### 💬 Chat History")
                
                # Show all conversations
                for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                    with st.container():
                        # User question
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%); 
                                    padding: 10px; border-radius: 10px; margin: 5px 0;">
                            <strong>You ({chat.get('timestamp', 'Now')}):</strong><br>
                            {chat['question']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # AI response
                        st.markdown(f"""
                        <div style="background: linear-gradient(90deg, #f3e5f5 0%, #e1bee7 100%); 
                                    padding: 10px; border-radius: 10px; margin: 5px 0 15px 0;">
                            <strong>🤖 AI Assistant:</strong><br>
                            {chat['answer']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Audio playback if available
                        if "audio_file" in chat and chat["audio_file"]:
                            try:
                                with open(chat["audio_file"], "rb") as f:
                                    audio_data = f.read()
                                st.audio(audio_data, format="audio/mp3")
                            except:
                                pass
        else:
            st.info("💡 Upload a PDF to start chatting!")
            st.markdown("""
            <div style="padding: 15px; background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%); border-radius: 10px; text-align: center; margin: 10px 0;">
                <strong>🚀 Quick Chat Features:</strong><br>
                • Type or speak questions<br>
                • Get instant answers<br>
                • Listen to responses<br>
                • View chat history
            </div>
            """, unsafe_allow_html=True)
        
        if uploaded_file is not None and not st.session_state.pdf_uploaded:
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
            if pdf_text:
                st.session_state.pdf_text = pdf_text
                st.success("✅ PDF text extracted successfully!")
                
                with st.spinner("Generating summary..."):
                    summary = summarize_text(pdf_text)
                    st.session_state.pdf_summary = summary
                    
                    # Document processed successfully
                    
                    st.session_state.pdf_uploaded = True
                    st.rerun()
            else:
                st.error("Failed to extract text from PDF. Please try another file.")
        
        if st.session_state.pdf_uploaded:
            st.success("✅ PDF loaded and summarized!")
            
            if st.button("🔄 Upload New PDF"):
                # Reset session state
                st.session_state.pdf_text = ""
                st.session_state.pdf_summary = ""
                st.session_state.chat_history = []
                st.session_state.pdf_uploaded = False
                st.rerun()
    

    
    # Main content area
    if st.session_state.pdf_uploaded:
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📋 Summary", "💬 Ask Questions"])
        
        with tab1:
            st.header("📋 Document Summary")
            
            # Create columns for summary and download
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.session_state.pdf_summary:
                    # Display summary with proper formatting
                    st.markdown("### 📝 Executive Summary")
                    
                    # Split summary into sections for better readability
                    summary_lines = st.session_state.pdf_summary.split('\n')
                    for line in summary_lines:
                        if line.strip():
                            if line.startswith('**') and line.endswith('**'):
                                # Display headers
                                st.markdown(f"##### {line}")
                            elif line.startswith('•'):
                                # Display bullet points
                                st.markdown(f"{line}")
                            else:
                                # Display regular text
                                st.markdown(line)
                else:
                    st.info("Upload a PDF to see the summary here.")
            
            with col2:
                # PDF Download Button
                if st.session_state.pdf_summary:
                    if st.button("📥 Download PDF Summary", use_container_width=True):
                        with st.spinner("Creating PDF..."):
                            pdf_buffer = create_summary_pdf(
                                st.session_state.pdf_text, 
                                st.session_state.pdf_summary,
                                None
                            )
                            if pdf_buffer:
                                st.download_button(
                                    label="📄 Download Summary.pdf",
                                    data=pdf_buffer,
                                    file_name="document_summary.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
            
            # Interactive Mindmap Section
            if st.session_state.pdf_text:
                st.markdown("---")
                st.markdown("### 🧠 Document Mindmap")
                st.markdown("Visual representation of key concepts and their relationships")
                
                with st.spinner("🧠 Creating AI-powered mindmap..."):
                    mindmap_fig = create_mindmap_with_ai(st.session_state.pdf_text)
                    if mindmap_fig:
                        st.plotly_chart(mindmap_fig, use_container_width=True)
                    else:
                        st.error("Unable to generate mindmap. Please try again.")
        
        with tab2:
            st.header("💬 Ask Questions")
            
            # Voice input section
            st.subheader("🎤 Voice Input")
            audio_bytes = audio_recorder(
                text="Click to record your question",
                recording_color="#e74c3c",
                neutral_color="#34495e",
                icon_name="microphone",
                icon_size="2x"
            )
            
            voice_question = ""
            if audio_bytes:
                st.info("🎵 Processing voice input...")
                voice_question = speech_to_text(audio_bytes)
                if voice_question:
                    st.success(f"🎤 Voice question: {voice_question}")
            
            # Text input section
            st.subheader("⌨️ Text Input")
            text_question = st.text_input(
                "Type your question here:",
                value=voice_question if voice_question else "",
                placeholder="What is this document about?"
            )
            
            # Ask question button
            if st.button("❓ Ask Question", disabled=not (text_question or voice_question)):
                question = text_question or voice_question
                
                with st.spinner("🤖 AI is finding answer..."):
                    answer = answer_question_with_ai(st.session_state.pdf_text, question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })
                    
                    # Generate audio for answer
                    audio_file = text_to_speech(answer)
                    
                    if audio_file:
                        st.session_state.chat_history[-1]["audio_file"] = audio_file
                    
                    st.rerun()
            
            # Chat history in the same tab
            if st.session_state.chat_history:
                st.markdown("---")
                st.header("💭 Conversation History")
                
                for i, chat in enumerate(reversed(st.session_state.chat_history)):
                    with st.expander(f"Q: {chat['question'][:50]}{'...' if len(chat['question']) > 50 else ''}", expanded=i==0):
                        st.markdown(f"**Question:** {chat['question']}")
                        st.markdown(f"**Answer:** {chat['answer']}")
                        
                        # Audio playback
                        if "audio_file" in chat and chat["audio_file"]:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                try:
                                    with open(chat["audio_file"], "rb") as audio_file:
                                        audio_bytes = audio_file.read()
                                    st.audio(audio_bytes, format="audio/mp3")
                                except Exception as e:
                                    st.error(f"Error playing audio: {str(e)}")
                            
                            with col2:
                                st.markdown(
                                    create_download_link(chat["audio_file"], "📥 Download Audio"),
                                    unsafe_allow_html=True
                                )
        

        

    
    else:
        # Welcome screen
        st.markdown(
            """
            ## Welcome to the PDF Summarizer & Q&A Chatbot! 🎉
            
            ### How to use this app:
            1. **📁 Upload a PDF** - Use the sidebar to upload your PDF document
            2. **📋 Read Summary** - Get an automatic summary of your document
            3. **💬 Ask Questions** - Use voice or text input to ask questions about the content
            4. **🔊 Listen to Answers** - Get spoken responses with text-to-speech
            
            ### Features:
            - 🤖 **Smart summarization** using advanced text analysis algorithms
            - 📊 **Basic document statistics** - word count, sentences, paragraphs
            - 📥 **Downloadable PDF summaries** with professional formatting
            - 💡 **Intelligent Q&A** using keyword matching and context extraction  
            - 🎤 **Voice input** for hands-free questions
            - 🔊 **Voice output** with downloadable audio responses
            - 📈 **Document statistics** with basic text analysis
            - 💾 **Conversation history** maintained during your session
            - 📱 **Responsive design** that works on different screen sizes
            - 🚀 **Works completely offline** - no API keys required!
            
            **Get started by uploading a PDF document using the sidebar!**
            """
        )

if __name__ == "__main__":
    main()
