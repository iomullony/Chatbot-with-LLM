import logging
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ─────────────────────────────────────────────
# 1. LOGGING SETUP
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
KB_PATH = PROJECT_ROOT / "knowledge_base.txt"
LOG_PATH = PROJECT_ROOT / "chatbot_history.log"

logging.basicConfig(
    filename=str(LOG_PATH),
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


def load_chunks(filepath: str | Path) -> list[str]:
    """
    Load knowledge_base.txt and split into chunks.
    Chunks are separated by blank lines; comment lines (starting with #) are skipped.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    paragraphs = raw.split("\n\n")
    chunks = []
    for para in paragraphs:
        lines = [line for line in para.splitlines() if not line.strip().startswith("#")]
        chunk = " ".join(lines).strip()
        if chunk:
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


def build_prompt(question: str, context_chunks: list[str], history: Optional[list[dict]] = None) -> str:
    """
    Build a prompt for flan-t5 using retrieved context and a short conversation history
    to better handle follow-up questions.
    """
    context = "\n".join(f"- {chunk}" for chunk in context_chunks)

    history_lines = []
    if history:
        for entry in history[-3:]:
            role = entry.get("role", "")
            content = str(entry.get("content", "")).strip()
            if role == "user" and content:
                history_lines.append(f"User: {content}")
            elif role == "assistant" and content:
                history_lines.append(f"Assistant: {content}")

    history_block = "\n".join(history_lines) if history_lines else "No previous conversation."

    prompt = (
        "You are a helpful assistant for the International Office at DIT (Deggendorf Institute of Technology). "
        "Answer the question based only on the context below. "
        "Use the conversation history only to resolve follow-up questions and references. "
        "If the answer is not in the context, say you don't have that information.\n\n"
        f"Conversation history:\n{history_block}\n\n"
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
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return retriever, tokenizer, model


@st.cache_resource(show_spinner="Indexing knowledge base...")
def load_knowledge_base(_retriever: SentenceTransformer):
    """
    Load chunks and pre-compute their embeddings.
    """
    chunks = load_chunks(KB_PATH)
    embeddings = _retriever.encode(chunks)
    return chunks, embeddings


# ─────────────────────────────────────────────
# 6. CORE: ANSWER GENERATION
# ─────────────────────────────────────────────


def generate_answer(question: str, chunks: list[str], chunk_embeddings: np.ndarray,
                    retriever: SentenceTransformer, tokenizer, model, history=None) -> tuple[str, list[str]]:
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, generate answer.
    Returns the answer text and the retrieved chunks (for display/debugging).
    """
    relevant_chunks = retrieve(question, chunks, chunk_embeddings, retriever, top_k=3)
    prompt = build_prompt(question, relevant_chunks, history=history)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, num_beams=4)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not answer:
        answer = "I don't have that information in the provided context."
    return answer, relevant_chunks


# ─────────────────────────────────────────────
# 7. STREAMLIT APP
# ─────────────────────────────────────────────


def initialize_runtime():
    """Load models and knowledge base once per session."""
    if "runtime_ready" not in st.session_state:
        with st.spinner("Loading models... this may take a minute on first run."):
            retriever, tokenizer, model = load_models()
            chunks, chunk_embeddings = load_knowledge_base(retriever)

        st.session_state.retriever = retriever
        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.chunks = chunks
        st.session_state.chunk_embeddings = chunk_embeddings
        st.session_state.runtime_ready = True


def main():
    st.set_page_config(page_title="THD/DIT International Office Chatbot", page_icon="🎓")
    st.title("🎓 THD/DIT International Office Chatbot")
    st.markdown("Ask me anything about exchange programs, applications, or studying at DIT.")

    initialize_runtime()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        logging.info("=" * 60)
        logging.info("NEW SESSION STARTED")
        logging.info("=" * 60)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources") and st.session_state.get("show_sources"):
                with st.expander("📄 Retrieved context chunks"):
                    for i, chunk in enumerate(message["sources"], 1):
                        st.caption(f"**Chunk {i}:** {chunk}")

    with st.sidebar:
        st.header("⚙️ Settings")
        st.session_state["show_sources"] = st.toggle("Show retrieved chunks", value=False)
        st.caption("Enabling this shows which parts of the knowledge base were used to generate each answer.")

    if prompt := st.chat_input("Type your question here..."):
        if not prompt.strip():
            st.warning("Please enter a question before sending.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        log_turn("user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = generate_answer(
                    prompt,
                    st.session_state.chunks,
                    st.session_state.chunk_embeddings,
                    st.session_state.retriever,
                    st.session_state.tokenizer,
                    st.session_state.model,
                    history=st.session_state.messages,
                )
            st.markdown(answer)

            if st.session_state.get("show_sources"):
                with st.expander("📄 Retrieved context chunks"):
                    for i, chunk in enumerate(sources, 1):
                        st.caption(f"**Chunk {i}:** {chunk}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
        log_turn("assistant", answer)


if __name__ == "__main__":
    main()
