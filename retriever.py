import json
import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

model = None
index = None
data = None


# LOAD RESOURCES 

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

   
    # LOAD  DATA
    

    with open(
        "data/assessments.json",
        "r",
        encoding="utf-8"
    ) as f:

        content = f.read()

  
    content = re.sub(
        r'[\x00-\x1F\x7F]',
        '',
        content
    )

    data = json.loads(content)

  
    # SEARCHABLE
  

    documents = []

    for item in data:

        text = f"""
        Assessment Name:
        {item.get('name', '')}

        Description:
        {item.get('description', '')}

        Job Levels:
        {' '.join(item.get('job_levels', []))}

        Categories:
        {' '.join(item.get('keys', []))}

        Languages:
        {' '.join(item.get('languages', []))}

        Remote:
        {item.get('remote', '')}

        Adaptive:
        {item.get('adaptive', '')}
        """

        documents.append(text)

   
    #  EMBEDDINGS
   
    print("Creating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=False
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

   
    # fAISS INDEX
   

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    print("FAISS index ready!")


# SEARCH 


def search_assessments(query, top_k=5):

    load_resources()

    query_lower = query.lower()

   
    # QUERY
   

    query_embedding = model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

  
    # VECTOR 
    

    distances, indices = index.search(
        query_embedding,
        top_k * 5
    )

    scored_results = []

   
    #  RANKING
  
    for rank, idx in enumerate(indices[0]):

        assessment = data[idx]

        score = 0

      
        # Semantic 
       

        score += (top_k * 5 - rank)

        
        # Searchable 
        

        searchable_text = f"""
        {assessment.get('name', '')}
        {assessment.get('description', '')}
        {' '.join(assessment.get('keys', []))}
        {' '.join(assessment.get('job_levels', []))}
        {' '.join(assessment.get('languages', []))}
        {assessment.get('remote', '')}
        """.lower()

        query_words = query_lower.split()

       
        # Keyword 
        

        for word in query_words:

            if len(word) <= 2:
                continue

            if word in searchable_text:

                score += 4

       
        # Technical Skill
        

        technical_keywords = [
            "java",
            "python",
            "developer",
            "backend",
            "frontend",
            "software",
            "coding",
            "cloud",
            "sql",
            "api"
        ]

        for keyword in technical_keywords:

            if (
                keyword in query_lower
                and keyword in searchable_text
            ):

                score += 8

       
        # Personality 
       
        personality_keywords = [
            "communication",
            "personality",
            "behavior",
            "leadership",
            "teamwork"
        ]

        for keyword in personality_keywords:

            if keyword in query_lower:

                if (
                    "personality" in searchable_text
                    or "competencies" in searchable_text
                ):

                    score += 8

       
        # Remote Work
       

        if "remote" in query_lower:

            if "yes" in assessment.get(
                "remote",
                ""
            ).lower():

                score += 5

        scored_results.append(
            (score, assessment)
        )

  
    #  RESULTS
    

    scored_results.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    
    # FINAL 
   

    final_results = []

    added_names = set()

    for score, assessment in scored_results:

        name = assessment.get("name")

        if name not in added_names:

            final_results.append({
                "name": assessment.get("name"),
                "url": assessment.get("link"),
                "description": assessment.get(
                    "description",
                    ""
                )
            })

            added_names.add(name)

        if len(final_results) >= top_k:

            break

    return final_results



def get_assessment_by_name(name):

    load_resources()

    for item in data:

        if (
            name.lower()
            in item.get("name", "").lower()
        ):

            return item

    return None
