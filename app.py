import streamlit as st
import numpy as np
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 1. LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="chatbot_history.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log_turn(role: str, text: str):
    """Append a single conversation turn to the log file."""
    logging.info(f"{role.upper()}: {text}")


# ─────────────────────────────────────────────
# 2. KNOWLEDGE BASE LOADING & CHUNKING
# ─────────────────────────────────────────────
def load_chunks(filepath: str) -> list[str]:
    """
    Load knowledge_base.txt and split into chunks.
    Chunks are separated by blank lines; comment lines (starting with #) are skipped.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on double newlines (paragraph boundaries)
    paragraphs = raw.split("\n\n")
    chunks = []
    for para in paragraphs:
        # Remove comment lines and strip whitespace
        lines = [l for l in para.splitlines() if not l.strip().startswith("#")]
        chunk = " ".join(lines).strip()
        if chunk:  # skip empty
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────
# 3. RAG: RETRIEVAL
# ─────────────────────────────────────────────
def retrieve(query: str, chunks: list[str], chunk_embeddings: np.ndarray,
             retriever: SentenceTransformer, top_k: int = 3) -> list[str]:
    """
    Encode the query, compute cosine similarity against all chunk embeddings,
    and return the top_k most relevant chunks.
    """
    query_embedding = retriever.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ─────────────────────────────────────────────
# 4. PROMPT BUILDING
# ─────────────────────────────────────────────
def build_prompt(question: str, context_chunks: list[str]) -> str:
    """
    Build a prompt for flan-t5 using the retrieved context chunks.
    flan-t5 works well with explicit instruction-style prompts.
    """
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)
    prompt = (
        f"You are a helpful assistant for the International Office at DIT (Deggendorf Institute of Technology). "
        f"Answer the question based only on the context below. "
        f"If the answer is not in the context, say you don't have that information.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt


# ─────────────────────────────────────────────
# 5. MODEL LOADING (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models... this may take a minute on first run.")
def load_models():
    """
    Load and cache both models so Streamlit doesn't reload them on every interaction.
    - retriever: sentence-transformers for RAG
    - tokenizer + model: flan-t5-small loaded directly (avoids pipeline task name issues)
    """
    retriever = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good quality
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return retriever, tokenizer, model


@st.cache_resource(show_spinner="Indexing knowledge base...")
def load_knowledge_base(_retriever: SentenceTransformer):
    """
    Load chunks and pre-compute their embeddings.
    Underscore prefix on _retriever tells Streamlit not to hash this argument.
    """
    chunks = load_chunks("knowledge_base.txt")
    embeddings = _retriever.encode(chunks)
    return chunks, embeddings


# ─────────────────────────────────────────────
# 6. CORE: ANSWER GENERATION
# ─────────────────────────────────────────────
def generate_answer(question: str, chunks: list[str], chunk_embeddings: np.ndarray,
                    retriever: SentenceTransformer, tokenizer, model) -> tuple[str, list[str]]:
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, generate answer.
    Returns the answer text and the retrieved chunks (for display/debugging).
    """
    relevant_chunks = retrieve(question, chunks, chunk_embeddings, retriever, top_k=3)
    prompt = build_prompt(question, relevant_chunks)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer, relevant_chunks


# ─────────────────────────────────────────────
# 7. STREAMLIT APP
# ─────────────────────────────────────────────
st.title("THD/DIT International Office Chatbot")
st.markdown("Ask me anything about exchange programs, applications, or studying at DIT.")

# Load models and knowledge base
retriever, tokenizer, model = load_models()
chunks, chunk_embeddings = load_knowledge_base(retriever)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Log session start
    logging.info("=" * 60)
    logging.info(f"NEW SESSION STARTED")
    logging.info("=" * 60)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Optionally show retrieved chunks for assistant messages (debug mode)
        if message["role"] == "assistant" and message.get("sources") and st.session_state.get("show_sources"):
            with st.expander("Retrieved context chunks"):
                for i, chunk in enumerate(message["sources"], 1):
                    st.caption(f"**Chunk {i}:** {chunk}")

# Sidebar: debug toggle
with st.sidebar:
    st.header("Settings")
    st.session_state["show_sources"] = st.toggle("Show retrieved chunks", value=False)
    st.caption("Enabling this shows which parts of the knowledge base were used to generate each answer.")

# Text input
if prompt := st.chat_input("Type your question here..."):

    # 1. Show and log user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    log_turn("user", prompt)

    # 2. Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = generate_answer(
                prompt, chunks, chunk_embeddings, retriever, tokenizer, model
            )
        st.markdown(answer)

        # Show sources if toggle is on
        if st.session_state.get("show_sources"):
            with st.expander("Retrieved context chunks"):
                for i, chunk in enumerate(sources, 1):
                    st.caption(f"**Chunk {i}:** {chunk}")

    # 3. Save and log assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
    log_turn("assistant", answer)