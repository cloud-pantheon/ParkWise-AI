from pathlib import Path
import re
import numpy as np

from pdf_loader import load_pdf
from chunker import chunk_pages


MODEL_NAME = "all-MiniLM-L6-v2"

PARK_ALIASES = {
    "redwood": "redwood.pdf",
    "redwood national": "redwood.pdf",
    "mount rainier": "mount_rainier.pdf",
    "rainier": "mount_rainier.pdf",
    "rocky mountain": "rocky_mountain.pdf",
    "rocky mountain national": "rocky_mountain.pdf",
}

TOPIC_EXPANSIONS = {
    "pets": "pets dogs allowed prohibited leash restrained trails animals",
    "camping": "camping campground campsite reservation backcountry wilderness permit",
    "hiking": "hiking trail trails route distance elevation difficulty conditions",
    "wildlife": "wildlife animals bear bears elk safety viewing distance",
    "safety": "safety warning danger emergency hazards weather preparedness",
    "permits": "permit permits reservation pass required rules regulations",
    "fees": "fee fees cost entrance pass payment",
}


def load_all_chunks():
    data_folder = Path(__file__).resolve().parent.parent / "data"
    pdf_files = list(data_folder.glob("*.pdf"))
    all_chunks = []

    for pdf_file in pdf_files:
        pages = load_pdf(pdf_file)
        chunks = chunk_pages(pages, chunk_size=250, overlap=50)
        all_chunks.extend(chunks)

    return all_chunks


def build_embeddings(chunks, model):
    texts = [chunk["text"] for chunk in chunks]
    return model.encode(texts, normalize_embeddings=True)


def detect_park_filters(query):
    query_lower = query.lower()
    detected = []

    for alias, filename in PARK_ALIASES.items():
        if alias in query_lower and filename not in detected:
            detected.append(filename)

    return detected


def expand_query(query):
    query_lower = query.lower()
    expansions = []

    keyword_map = {
        "pets": ("pet", "pets", "dog", "dogs"),
        "camping": ("camp", "camping", "campground", "campsite"),
        "hiking": ("hike", "hiking", "trail", "trails"),
        "wildlife": ("wildlife", "animal", "animals", "bear", "elk"),
        "safety": ("safe", "safety", "danger", "hazard", "emergency"),
        "permits": ("permit", "permits", "reservation", "required"),
        "fees": ("fee", "fees", "cost", "price", "entrance"),
    }

    for topic, terms in keyword_map.items():
        if any(term in query_lower for term in terms):
            expansions.append(TOPIC_EXPANSIONS[topic])

    if not expansions:
        return query

    return f"{query} {' '.join(expansions)}"


def search(
    query,
    chunks,
    embeddings,
    model,
    top_k=5,
    park_filters=None,
    use_auto_park_filter=True,
):
    """Hybrid semantic retrieval controllable by the agent.

    park_filters can be a filename, a list of filenames, or None. When None,
    the retriever can optionally infer park filters from the query.
    """
    if isinstance(park_filters, str):
        park_filters = [park_filters]

    if park_filters is None and use_auto_park_filter:
        detected = detect_park_filters(query)
        park_filters = detected or None

    expanded_query = expand_query(query)
    query_embedding = model.encode(expanded_query, normalize_embeddings=True)
    scores = np.dot(embeddings, query_embedding)

    query_words = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
    stop_words = {
        "the", "a", "an", "is", "are", "can", "i", "we", "you", "to",
        "of", "in", "on", "at", "for", "and", "or", "what", "which",
        "how", "does", "do", "about", "with", "my", "me", "there"
    }
    meaningful_query_words = query_words - stop_words

    candidate_results = []

    for index, score in enumerate(scores):
        chunk = chunks[index]

        if park_filters and chunk["source"] not in park_filters:
            continue

        adjusted_score = float(score)
        chunk_words = set(re.findall(r"\b[a-zA-Z]+\b", chunk["text"].lower()))

        # Lightweight lexical boost. Semantic similarity remains dominant.
        lexical_hits = len(meaningful_query_words.intersection(chunk_words))
        adjusted_score += min(lexical_hits, 5) * 0.025

        candidate_results.append({
            "score": adjusted_score,
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        })

    candidate_results.sort(key=lambda x: x["score"], reverse=True)
    return candidate_results[:top_k]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    print("Loading ParkWise documents...")
    chunks = load_all_chunks()
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = build_embeddings(chunks, model)

    print("\nParkWise Semantic Search is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Ask a question: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break

        results = search(query, chunks, embeddings, model, top_k=5)
        print("\nTop Results:")
        print("=" * 70)

        for number, result in enumerate(results, start=1):
            print(f"\nResult {number}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Source: {result['source']}")
            print(f"Page: {result['page']}")
            print(f"Chunk: {result['chunk_id']}")
            print("\nText:")
            print(result["text"][:700])
            print("\n" + "-" * 70)
