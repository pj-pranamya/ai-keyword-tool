from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
import re
import requests
import time
import spacy
nlp = spacy.load("en_core_web_sm")


# ---------------------------
# CONFIG
# ---------------------------

STOP_WORDS = {
    "using", "based", "approach", "method", "analysis", "study",
    "system", "model", "technique", "paper", "research", "work"
}

DOMAIN_KEYWORDS = {"alzheimer", "dementia", "cognitive", "neuro", "speech"}


# ---------------------------
# CLEANING FUNCTIONS
# ---------------------------

def clean_keywords(keywords):
    cleaned = []

    for word, score in keywords:
        word_lower = word.lower().strip()

        if len(word_lower) < 4:
            continue

        if any(stop in word_lower.split() for stop in STOP_WORDS):
            continue

        cleaned.append(word_lower)

    final_keywords = []
    for word in cleaned:
        if not any(word != other and word in other for other in cleaned):
            final_keywords.append(word)

    return list(set(final_keywords))


def refine_keywords(keywords):
    refined = []

    keywords = sorted(keywords, key=len, reverse=True)

    for kw in keywords:
        if not any(kw in other and kw != other for other in refined):
            refined.append(kw)

    return refined


def filter_phrases(phrases):
    bad_starts = {"of", "for", "and", "in", "on", "by", "with", "to"}
    bad_ends = {"of", "for", "and", "in", "on", "by", "with", "to"}

    cleaned = []

    for phrase in phrases:
        words = phrase.split()

        if len(words) < 2:
            continue

        if words[0] in bad_starts:
            continue

        if words[-1] in bad_ends:
            continue

        # remove broken long tokens
        if any(len(w) > 20 for w in words):
            continue

        cleaned.append(phrase)

    return list(set(cleaned))


def rank_keywords_by_relevance(keywords, original_text):
    if not keywords:
        return keywords

    text_embedding = semantic_model.encode(original_text, convert_to_tensor=True)
    keyword_embeddings = semantic_model.encode(keywords, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]

    scored = list(zip(keywords, similarities.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [kw for kw, score in scored]


def anchor_filter(terms, root_term):
    root_words = set(root_term.lower().split())
    filtered = []

    for term in terms:
        term_words = set(term.split())

        # Overlap with root words
        if root_words & term_words:
            filtered.append(term)
            continue

        # OR domain related
        if term_words & DOMAIN_KEYWORDS:
            filtered.append(term)

    return filtered


# ---------------------------
# API FETCHING
# ---------------------------

def fetch_dynamic_titles(query):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": 35,
        "fields": "title,abstract"
    }

    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()

                documents = []

                for paper in data.get("data", []):
                    if paper.get("title"):
                        documents.append(paper["title"])

                    if paper.get("abstract"):
                        documents.append(paper["abstract"])

                if documents:
                    return documents

        except Exception as e:
            print("Semantic Scholar failed:", e)

        time.sleep(1)

    print("Semantic Scholar returned nothing. Trying OpenAlex...")
    return fetch_openalex_titles(query)


def fetch_openalex_titles(query):
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per_page": 20
    }

    try:
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return [work["title"] for work in data.get("results", []) if work.get("title")]

    except Exception as e:
        print("OpenAlex failed:", e)

    return []


# ---------------------------
# EXPANSION LOGIC
# ---------------------------

def expand_term_semantically(term):

    titles = fetch_dynamic_titles(term)

    if not titles:
        return []

    concept_bank = []

    for title in titles:
    # extract keyphrases from each title
     extracted = kw_model.extract_keywords(
        title,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=8
    )

    for phrase, score in extracted:
        concept_bank.append(phrase.lower())

    concept_bank = list(set(concept_bank))

    if not concept_bank:
        return []

    term_embedding = semantic_model.encode(term, convert_to_tensor=True)
    bank_embeddings = semantic_model.encode(concept_bank, convert_to_tensor=True)

    similarities = util.cos_sim(term_embedding, bank_embeddings)[0]

    scored_terms = []

    for i, score in enumerate(similarities):
        if score > 0.60:
            scored_terms.append((concept_bank[i], score.item()))

    scored_terms.sort(key=lambda x: x[1], reverse=True)

    expanded_terms = [term for term, score in scored_terms]

    expanded_terms = anchor_filter(expanded_terms, term)
    expanded_terms = filter_phrases(expanded_terms)
    expanded_terms = clean_keywords([(t, 1.0) for t in expanded_terms])
    expanded_terms = refine_keywords(expanded_terms)
    expanded_terms = pos_filter_phrases(expanded_terms)
    expanded_terms = rank_keywords_by_relevance(expanded_terms, term)

    return expanded_terms[:15]


# ---------------------------
# BOOLEAN BUILDER
# ---------------------------

def build_boolean_query(or_groups, not_group=None):
    query_parts = []

    for group in or_groups:
        if group:
            or_part = " OR ".join([f'"{term}"' for term in group])
            query_parts.append(f"({or_part})")

    final_query = " AND ".join(query_parts)

    if not_group:
        not_part = " OR ".join([f'"{term}"' for term in not_group])
        final_query += f" NOT ({not_part})"

    return final_query

def pos_filter_phrases(phrases):
    filtered = []

    for phrase in phrases:
        doc = nlp(phrase)

        # Only allow phrases that are mostly noun-based
        pos_tags = [token.pos_ for token in doc]

        # Accept patterns like:
        # ADJ NOUN
        # NOUN NOUN
        # ADJ NOUN NOUN
        # NOUN NOUN NOUN

        if all(pos in {"NOUN", "PROPN", "ADJ"} for pos in pos_tags):
            filtered.append(phrase)

    return filtered


# ---------------------------
# FLASK APP
# ---------------------------

app = Flask(__name__)
CORS(app)

kw_model = KeyBERT()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route("/")
def home():
    return render_template("index.html")

CORS(app)

socketio = SocketIO(app, cors_allowed_origins='*')

kw_model = KeyBERT()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')


@app.route("/keywords", methods=["POST"])
def generate_keywords():
    data = request.json
    text = data.get("text", "")

    # Step 1: Extract from user input
    base_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=8
    )

    base_keywords = [kw for kw, score in base_keywords]

    # Step 2: Fetch research docs
    documents = fetch_dynamic_titles(text)

    research_keywords = []

    for doc in documents:
        extracted = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            top_n=5
        )

        for phrase, score in extracted:
            research_keywords.append(phrase.lower())

    # Step 3: Merge
    all_keywords = list(set(base_keywords + research_keywords))

    # Step 4: Clean
    all_keywords = filter_phrases(all_keywords)
    all_keywords = clean_keywords([(k, 1.0) for k in all_keywords])
    all_keywords = refine_keywords(all_keywords)
    all_keywords = pos_filter_phrases(all_keywords)

    # Step 5: Rank semantically
    # Semantic ranking
    text_embedding = semantic_model.encode(text, convert_to_tensor=True)
    keyword_embeddings = semantic_model.encode(all_keywords, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, keyword_embeddings)[0]

    scored = list(zip(all_keywords, similarities.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

# Tier 1: High precision core
    core = [kw for kw, score in scored if score > 0.65][:12]

# Tier 2: Broader but still relevant
    explore = [kw for kw, score in scored if 0.55 < score <= 0.65][:20]

    clusters = cluster_keywords(final_keywords)
    # Emit suggestions + activity to connected realtime clients
    try:
        socketio.emit('suggestions', {'keywords': final_keywords, 'clusters': clusters})
        socketio.emit('activity', {'msg': f'Generated {len(final_keywords)} keywords'})
    except Exception:
        pass

    return jsonify({
        "clusters": clusters,
        "keywords": final_keywords
    })

@app.route("/expand-term", methods=["POST"])
def expand_term():
    data = request.json
    term = data.get("term", "").strip()

    if not term:
        return jsonify({"expanded": []})

    print(f"\nExpanding term: {term}")

    expanded_terms = expand_term_semantically(term)

    print("Expanded terms:", expanded_terms[:10])
    # Emit incremental suggestions and activity
    try:
        socketio.emit('suggestions_append', {'expanded': expanded_terms[:15]})
        socketio.emit('activity', {'msg': f"Expanded '{term}' → {len(expanded_terms[:15])} terms"})
    except Exception:
        pass

    return jsonify({
        "expanded": expanded_terms
    })


@app.route("/build-query", methods=["POST"])
def build_query_route():
    data = request.json

    or_groups = data.get("or_groups", [])
    not_group = data.get("not_group", [])

    boolean_query = build_boolean_query(or_groups, not_group)
    try:
        socketio.emit('activity', {'msg': 'Built boolean query'})
        socketio.emit('boolean_query', {'query': boolean_query})
    except Exception:
        pass

    return jsonify({
        "boolean_query": boolean_query
    })


if __name__ == "__main__":
    # Use SocketIO runner (eventlet recommended)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)

