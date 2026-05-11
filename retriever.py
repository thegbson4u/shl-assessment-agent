import json
import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

# ==================================================
# GLOBAL VARIABLES
# ==================================================

model = None
index = None
data = None

# ==================================================
# LOAD RESOURCES ONLY WHEN NEEDED
# ==================================================

def load_resources():

    global model
    global index
    global data

    # Prevent reloading
    if model is not None:
        return

    print("Loading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

    # -----------------------------
    # LOAD JSON
    # -----------------------------

    with open("data/assessments.json", "r", encoding="utf-8") as f:

        content = f.read()

    content = re.sub(r'[\x00-\x1F\x7F]', '', content)

    data = json.loads(content)

    # -----------------------------
    # CREATE DOCUMENTS
    # -----------------------------

    documents = []

    for item in data:

        text = f"""
        Assessment Name: {item.get('name', '')}

        Description:
        {item.get('description', '')}

        Job Levels:
        {' '.join(item.get('job_levels', []))}

        Categories:
        {' '.join(item.get('keys', []))}
        """

        documents.append(text)

    # -----------------------------
    # CREATE EMBEDDINGS
    # -----------------------------

    print("Creating embeddings...")

    embeddings = model.encode(documents)

    embeddings = np.array(embeddings).astype("float32")

    # -----------------------------
    # CREATE FAISS INDEX
    # -----------------------------

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS ready!")

# ==================================================
# SEARCH FUNCTION
# ==================================================

def search_assessments(query, top_k=5):

    load_resources()

    query_lower = query.lower()

    query_embedding = model.encode([query])

    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k * 3)

    scored_results = []

    for rank, idx in enumerate(indices[0]):

        assessment = data[idx]

        score = 0

        score += (top_k * 3 - rank)

        searchable_text = f"""
        {assessment.get('name', '')}
        {assessment.get('description', '')}
        {' '.join(assessment.get('keys', []))}
        {' '.join(assessment.get('job_levels', []))}
        """.lower()

        query_words = query_lower.split()

        for word in query_words:

            if word in searchable_text:

                score += 3

        scored_results.append((score, assessment))

    scored_results.sort(reverse=True, key=lambda x: x[0])

    final_results = []

    added_names = set()

    for score, assessment in scored_results:

        name = assessment.get("name")

        if name not in added_names:

            final_results.append({
                "name": assessment.get("name"),
                "url": assessment.get("link"),
                "description": assessment.get("description")
            })

            added_names.add(name)

        if len(final_results) >= top_k:

            break

    return final_results

# ==================================================
# COMPARISON FUNCTION
# ==================================================

def get_assessment_by_name(name):

    load_resources()

    for item in data:

        if name.lower() in item.get("name", "").lower():

            return item

    return None
