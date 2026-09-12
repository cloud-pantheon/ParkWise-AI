import os
import sys
import streamlit as st

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from retriever import load_all_chunks, build_embeddings
from agentic_rag import run_agentic_rag


MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="ParkWise Agentic AI",
    page_icon="🏞️",
    layout="centered",
)

SOURCE_NAMES = {
    "redwood.pdf": "Redwood National & State Parks",
    "mount_rainier.pdf": "Mount Rainier National Park",
    "rocky_mountain.pdf": "Rocky Mountain National Park",
}


def pretty_source(filename):
    return SOURCE_NAMES.get(filename, filename)


def unique_page_sources(results):
    if not results:
        return []

    best_score = results[0]["score"]
    seen = set()
    sources = []

    for result in results:
        if result["score"] < best_score - 0.20:
            continue

        key = (result["source"], result["page"])
        if key in seen:
            continue

        seen.add(key)
        sources.append({
            "source": result["source"],
            "page": result["page"],
            "score": result["score"],
        })

    return sources


def render_sources(sources):
    for source in sources:
        st.markdown(
            f"""
**🌿 {pretty_source(source['source'])}**  
📄 Page **{source['page']}**  
🎯 Relevance **{source['score']:.2f}**
"""
        )
        st.divider()


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.write("API key detected:", bool(api_key))

if not api_key:
    st.error("Gemini API key was not found in the environment variables.")
    st.stop()

client = genai.Client(api_key=api_key)


@st.cache_resource
def initialize_knowledge_base():
    chunks = load_all_chunks()
    embedding_model = SentenceTransformer(MODEL_NAME)
    embeddings = build_embeddings(chunks, embedding_model)
    return chunks, embedding_model, embeddings


with st.spinner("Loading ParkWise knowledge base..."):
    chunks, embedding_model, embeddings = initialize_knowledge_base()


st.title("🏞️ ParkWise Agentic RAG")
st.markdown(
    "Ask questions about U.S. National Parks. ParkWise now uses an AI agent "
    "to plan retrieval, evaluate evidence, and retry searches when needed."
)
st.info(
    "Agent workflow: Plan → Retrieve → Evaluate → Replan if needed → Grounded answer"
)


with st.sidebar:
    st.header("🤖 ParkWise Agent")
    st.write(
        "An Agentic RAG system that decides how to search its National Park "
        "knowledge base instead of using a single fixed retrieval pass."
    )

    st.divider()
    st.subheader("Available Parks")
    st.write("🌲 Redwood National & State Parks")
    st.write("🌋 Mount Rainier National Park")
    st.write("🏔️ Rocky Mountain National Park")

    st.divider()
    st.subheader("Knowledge Base")
    st.metric("Document chunks", len(chunks))
    st.caption("Embedding model: all-MiniLM-L6-v2")

    st.divider()
    debug_mode = st.toggle("🧠 Show Agent Decision Trace")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if debug_mode and message.get("trace"):
                with st.expander("🧠 Agent Decision Trace"):
                    for index, step in enumerate(message["trace"], start=1):
                        st.markdown(f"**{index}. {step['stage']}**")
                        st.write(step["detail"])

                    if message.get("attempts"):
                        st.caption(
                            f"Retrieval attempts: {message['attempts']} · "
                            f"Evidence confidence: {message.get('confidence', 'unknown').title()}"
                        )

            if message.get("sources"):
                with st.expander("📚 View Sources"):
                    render_sources(message["sources"])


question = st.chat_input(
    "Ask about pets, hiking, camping, wildlife, permits, safety..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent is planning and searching park documents..."):
            result = run_agentic_rag(
                question=question,
                chunks=chunks,
                embeddings=embeddings,
                embedding_model=embedding_model,
                client=client,
            )

        st.markdown(result.answer)

        trace_data = [
            {"stage": step.stage, "detail": step.detail}
            for step in result.trace
        ]

        if debug_mode:
            with st.expander("🧠 Agent Decision Trace", expanded=True):
                for index, step in enumerate(result.trace, start=1):
                    st.markdown(f"**{index}. {step.stage}**")
                    st.write(step.detail)

                st.caption(
                    f"Retrieval attempts: {result.attempts} · "
                    f"Evidence confidence: {result.confidence.title()}"
                )

        sources = unique_page_sources(result.sources)

        if sources:
            with st.expander("📚 View Sources"):
                render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "sources": sources,
        "trace": trace_data,
        "attempts": result.attempts,
        "confidence": result.confidence,
    })
