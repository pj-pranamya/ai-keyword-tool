from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
"using","based","approach","method","analysis","study",
"system","model","technique","paper","research","work",
"improve","improved","improving","background","abstract",
"novel","framework","algorithm","towards","via","using"
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
        if " ad " in word_lower:
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
        # remove weird noun fragments
        if words[-1] in {"brains", "things", "stuff"}:
            continue
        # remove short meaningless phrases
        if len(words) == 2 and words[1] in {"act", "thing", "type"}:
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

def cluster_keywords(keywords, text):

    if not keywords:
        return {
            "problem_terms": [],
            "method_terms": [],
            "data_terms": [],
            "other_terms": []
        }

    embeddings = semantic_model.encode(keywords, convert_to_tensor=True)
    text_embedding = semantic_model.encode(text, convert_to_tensor=True)

    similarities = util.cos_sim(text_embedding, embeddings)[0]

    scored = list(zip(keywords, similarities.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    problem_terms = [kw for kw, s in scored[:5]]
    method_terms = [kw for kw, s in scored[5:10]]
    data_terms = [kw for kw, s in scored[10:15]]
    other_terms = [kw for kw, s in scored[15:]]

    return {
        "problem_terms": problem_terms,
        "method_terms": method_terms,
        "data_terms": data_terms,
        "other_terms": other_terms
    }


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
            documents = []

            for work in data.get("results", []):
             if work.get("title"):
              documents.append(work["title"])

             if work.get("abstract_inverted_index"):
              abstract_words = []
              for word, positions in work["abstract_inverted_index"].items():
               abstract_words.append(word)
              documents.append(" ".join(abstract_words))

            return documents

    except Exception as e:
        print("OpenAlex failed:", e)

    return []


# ---------------------------
# EXPANSION LOGIC
# ---------------------------

def expand_term_semantically(term):

    query = term + " " + ORIGINAL_QUERY
    titles = fetch_dynamic_titles(query)

    if not titles:
        return []

    concept_bank = []

    for title in titles:

     doc = nlp(title)

     for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()

        if len(phrase.split()) >= 2:
            concept_bank.append(phrase)
    concept_bank = list(set(concept_bank))

    if not concept_bank:
        return []

    term_embedding = semantic_model.encode(term, convert_to_tensor=True)
    bank_embeddings = semantic_model.encode(concept_bank, convert_to_tensor=True)

    similarities = util.cos_sim(term_embedding, bank_embeddings)[0]

    scored_terms = []

    for i, score in enumerate(similarities):
        if score > 0.48:
            scored_terms.append((concept_bank[i], score.item()))

    scored_terms.sort(key=lambda x: x[1], reverse=True)

    expanded_terms = [term for term, score in scored_terms]

    expanded_terms = anchor_filter(expanded_terms, term)
    expanded_terms = filter_phrases(expanded_terms)
    expanded_terms = clean_keywords([(t, 1.0) for t in expanded_terms])
    expanded_terms = refine_keywords(expanded_terms)
    #expanded_terms = pos_filter_phrases(expanded_terms)
    expanded_terms = rank_keywords_by_relevance(expanded_terms, term)

    diverse_terms = []

    for term in expanded_terms:

     term_embedding = semantic_model.encode(term, convert_to_tensor=True)

     duplicate = False

     for existing in diverse_terms:
        existing_embedding = semantic_model.encode(existing, convert_to_tensor=True)

        sim = util.cos_sim(term_embedding, existing_embedding).item()

        if sim > 0.75:
            duplicate = True
            break

     if not duplicate:
        diverse_terms.append(term)

    expanded_terms = diverse_terms

    return expanded_terms[:25]


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

        if "NOUN" in pos_tags and pos_tags[-1] in {"NOUN", "PROPN"}:
            filtered.append(phrase)

    return filtered


# ---------------------------
# FLASK APP
# ---------------------------

app = Flask(__name__)
CORS(app)

kw_model = KeyBERT()
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

def remove_similar_keywords(keywords):

    filtered = []

    for kw in keywords:

        if not any(kw in other or other in kw for other in filtered):
            filtered.append(kw)

    return filtered


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/keywords", methods=["POST"])
def generate_keywords():
    print("ROUTE HIT")
    data = request.json
    text = data.get("text", "")
    global ORIGINAL_QUERY
    ORIGINAL_QUERY = text

    # Step 1: Extract from user input
    base_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 3),
        stop_words='english',
        top_n=40
    )

    base_keywords = [kw for kw, score in base_keywords]

    # Step 2: Fetch research docs
    documents = fetch_dynamic_titles(text)

    research_keywords = []

    for doc in documents:

     parsed = nlp(doc)

     for chunk in parsed.noun_chunks:

        phrase = chunk.text.lower().strip()

        if 2 <= len(phrase.split()) <= 4:
            research_keywords.append(phrase)
    print("Research keywords:", len(research_keywords))
    # Step 3: Merge
    all_keywords = list(set(base_keywords + research_keywords))
    unique = []
    seen = set()

    for kw in all_keywords:
     key = kw.lower().replace("'s","")
     if key not in seen:
        unique.append(kw)
        seen.add(key)

    all_keywords = unique
    all_keywords = sorted(all_keywords, key=len)
    

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
    #core = [kw for kw, score in scored if score > 0.65][:12]

# Tier 2: Broader but still relevant
    #explore = [kw for kw, score in scored if 0.55 < score <= 0.65][:20]

    #candidates = list(dict.fromkeys(core + explore))
    #final_keywords = candidates
    final_keywords = [kw for kw, score in scored[:30]]
    


    clusters_data = cluster_keywords(final_keywords, text)

    flat_clusters = {}

    for kw in final_keywords:

     term_embedding = semantic_model.encode(kw, convert_to_tensor=True)
     keyword_embeddings = semantic_model.encode(final_keywords, convert_to_tensor=True)

     similarities = util.cos_sim(term_embedding, keyword_embeddings)[0]

     scored = list(zip(final_keywords, similarities.tolist()))
     scored.sort(key=lambda x: x[1], reverse=True)

     related = []

    for k, s in scored:
     if k != kw and s > 0.55:
        related.append(k)

     related = related[:5]

     flat_clusters[kw] = related

    return jsonify({
    "keywords": final_keywords,
    "clusters": flat_clusters
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

    return jsonify({
        "expanded": expanded_terms
    })


@app.route("/build-query", methods=["POST"])
def build_query_route():
    data = request.json

    or_groups = data.get("or_groups", [])
    not_group = data.get("not_group", [])

    boolean_query = build_boolean_query(or_groups, not_group)

    return jsonify({
        "boolean_query": boolean_query
    })

@app.route("/definition", methods=["POST"])
def get_definition():

    data = request.json
    term = data.get("term","")

    try:

        # try whole phrase first
        url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}"
        r = requests.get(url)

        if r.status_code == 200:
            definition = r.json()[0]["meanings"][0]["definitions"][0]["definition"]
            return jsonify({"definition":definition})

        # fallback: explain first meaningful word
        words = term.split()

        for w in words:
            if len(w) > 3:
                url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{w}"
                r = requests.get(url)

                if r.status_code == 200:
                    definition = r.json()[0]["meanings"][0]["definitions"][0]["definition"]
                    return jsonify({"definition":definition})

    except:
        pass

    return jsonify({"definition":"Research concept related to: " + term})

if __name__ == "__main__":
    app.run(debug=True)