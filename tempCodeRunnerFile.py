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
try:
    from google import genai
    from google.genai import types