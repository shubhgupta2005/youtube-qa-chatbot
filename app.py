import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from openai import OpenAI
import faiss
import numpy as np
from urllib.parse import urlparse, parse_qs
import re

# It's a good practice to handle the API key securely.
# This code assumes you have set up your OpenAI key in Streamlit's secrets.
try:
    client = OpenAI()
except Exception:
    st.error("OpenAI API key not found. Please set it up in your Streamlit secrets.")
    st.stop()

# ----------------------
# Step 1: Extract video ID
# ----------------------
def extract_video_id(youtube_url: str):
    """Extracts the video ID from various YouTube URL formats."""
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        video_id = parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.hostname == 'youtu.be':
        video_id = parsed_url.path.lstrip('/')
    else:
        # Handles embed URLs
        match = re.match(r'/embed/([a-zA-Z0-9_-]+)', parsed_url.path)
        video_id = match.group(1) if match else None
    return video_id

# ----------------------
# Step 2: Get transcript
# ----------------------
def get_transcript_from_url(youtube_url: str):
    """Fetches the transcript for a given YouTube URL."""
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return None
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        transcript_text = " ".join([t['text'] for t in transcript_list if t['text'].strip()])
        return transcript_text
    except TranscriptsDisabled:
        st.error(f"Transcripts are disabled for video: {video_id}")
        return None
    except Exception as e:
        st.error(f"Could not retrieve transcript for video: {video_id}. Error: {e}")
        return None

# ----------------------
# Step 3: Chunk text
# ----------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Splits the text into overlapping chunks."""
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
    """Creates embeddings for text chunks and builds a FAISS index."""
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
    """Searches for the most relevant chunks based on a query."""
    response = client.embeddings.create(model=model, input=[query])
    q_embed = np.array(response.data[0].embedding, dtype="float32").reshape(1, -1)
    _, indices = index.search(q_embed, top_k)
    return [chunks[i] for i in indices[0]]

def answer_question(query: str, index, chunks: list, top_k: int = 3, model: str = "gpt-4o-mini"):
    """Generates an answer to a question using context from relevant chunks."""
    relevant_chunks = search_chunks(query, index, chunks, top_k)
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Answer the question based only on the following transcript context.
If the answer is not available in the context, say "I couldn't find the answer in the transcript."

Context:
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
st.info("Paste a YouTube link, process the video, and then ask your questions!")

# Initialize session state variables if they don't exist
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
    st.session_state.faiss_index = None
    st.session_state.chunks = None
    st.session_state.youtube_url = ""

youtube_url = st.text_input("Paste your YouTube video link here:", key="youtube_url_input")

if st.button("Process Video"):
    if youtube_url:
        st.session_state.youtube_url = youtube_url
        with st.spinner("Processing video... This may take a moment."):
            transcript = get_transcript_from_url(youtube_url)
            if transcript:
                chunks = chunk_text(transcript)
                index, stored_chunks = create_embeddings(chunks)
                
                # Store processed data in session state
                st.session_state.faiss_index = index
                st.session_state.chunks = stored_chunks
                st.session_state.video_processed = True
                st.success("Video processed successfully! You can now ask questions below.")
            else:
                st.session_state.video_processed = False # Reset on failure
    else:
        st.warning("Please paste a YouTube URL first.")

# Only show the question input if a video has been processed successfully
if st.session_state.video_processed:
    st.header(f"Ask a question about the video")
    question = st.text_input("Enter your question:", key="question_input")

    if question:
        with st.spinner("Searching for the answer..."):
            index = st.session_state.faiss_index
            stored_chunks = st.session_state.chunks
            
            answer = answer_question(question, index, stored_chunks)
            st.write("### Answer")
            st.success(answer)
