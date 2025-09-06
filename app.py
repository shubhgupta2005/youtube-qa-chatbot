%%writefile app.py
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from openai import OpenAI
import faiss
import numpy as np
from urllib.parse import urlparse, parse_qs
import re

# Initialize OpenAI client
client = OpenAI()

# ----------------------
# Step 1: Extract video ID
# ----------------------
def extract_video_id(youtube_url: str):
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path.lstrip('/')
    match = re.match(r'/embed/([a-zA-Z0-9_-]+)', parsed_url.path)
    if match:
        return match.group(1)
    return None

# ----------------------
# Step 2: Get transcript
# ----------------------
def get_transcript_from_url(youtube_url: str):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript_text = " ".join([t['text'] for t in transcript_list if t['text'].strip()])
        return transcript_text
    except TranscriptsDisabled:
        return None
    except Exception:
        return None

# ----------------------
# Step 3: Chunk text
# ----------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    chunks, start, length = [], 0, len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ----------------------
# Step 4: Embeddings + FAISS
# ----------------------
def create_embeddings(chunks: list, model: str = "text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=chunks)
    embeddings = [np.array(e.embedding, dtype="float32") for e in response.data]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, chunks

# ----------------------
# Step 5: Search & Answer
# ----------------------
def search_chunks(query: str, index, chunks: list, top_k: int = 3, model: str = "text-embedding-3-small"):
    response = client.embeddings.create(model=model, input=query)
    q_embed = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(q_embed, top_k)
    return [chunks[i] for i in indices[0]]

def answer_question(query: str, index, chunks: list, top_k: int = 3, model: str = "gpt-4o-mini"):
    relevant_chunks = search_chunks(query, index, chunks, top_k)
    context = "\n\n".join(relevant_chunks)
    prompt = f"""Answer the question using the following transcript context:

{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for YouTube videos."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ----------------------
# Streamlit UI
# ----------------------
st.title("🎥 YouTube Q&A Chatbot")

youtube_url = st.text_input("Paste your YouTube video link here:")
question = st.text_input("Ask a question about the video:")

if youtube_url and question:
    transcript = get_transcript_from_url(youtube_url)
    if transcript:
        chunks = chunk_text(transcript)
        index, stored_chunks = create_embeddings(chunks)
        answer = answer_question(question, index, stored_chunks)
        st.success(answer)
    else:
        st.error("Transcript not available for this video")
