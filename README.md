# Conversational SHL Assessment Recommender

### Prince Kumar Maurya

B.Tech CSE Student | AI/ML Enthusiast
GitHub: [https://github.com/thegbson4u/shl-assessment-agent](https://github.com/thegbson4u/shl-assessment-agent)
Live API: [https://shl-assessment-agent-production-f23a.up.railway.app](https://shl-assessment-agent-production-f23a.up.railway.app)

---

## Overview

This project is a conversational SHL assessment recommendation system built using FastAPI. It helps recruiters find relevant SHL assessments through natural language queries instead of manual catalog searching.

Users can describe:

* hiring role,
* technical skills,
* personality traits,
* experience level,
* and hiring requirements.

The system returns grounded SHL assessment recommendations directly from the official catalog.

---

## Approach

I used a retrieval-based architecture to ensure reliable and valid recommendations.

The SHL catalog data was cleaned and converted into searchable documents containing:

* assessment names,
* descriptions,
* job levels,
* and competency information.

SentenceTransformers (`all-MiniLM-L6-v2`) was used for semantic embeddings, while FAISS handled vector similarity search.

A hybrid ranking system combining semantic similarity and keyword boosting was implemented to improve recommendation relevance.

---

## Features

The system supports:

* clarification for vague queries,
* recommendation refinement,
* assessment comparison,
* and off-topic refusal.

Examples:

* vague hiring queries trigger follow-up questions,
* new requirements update recommendations dynamically,
* comparisons like “OPQ vs GSA” are supported.

---

## Deployment

Initially, deployment on Render caused memory issues because transformer models loaded during startup. This was optimized using:

* lazy loading,
* CPU-only inference,
* and Railway deployment.

---

## Technologies Used

* Python
* FastAPI
* SentenceTransformers
* FAISS
* Railway
* GitHub

AI-assisted tools were used for debugging, deployment troubleshooting, and retrieval optimization.

---

## Final Thoughts

This project helped me understand semantic search, conversational APIs, vector databases, and ML deployment workflows.

The final system focuses on reliability, relevance, and grounded conversational recommendations.
