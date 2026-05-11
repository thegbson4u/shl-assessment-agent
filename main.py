from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from retriever import (
    search_assessments,
    get_assessment_by_name
)

app = FastAPI()


# ==================================================
# REQUEST MODELS
# ==================================================

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


# ==================================================
# HEALTH ENDPOINT
# ==================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.post("/chat")
def chat(request: ChatRequest):

    latest_message = request.messages[-1].content.lower()

    # ==================================================
    # BUILD FULL CONVERSATION CONTEXT
    # ==================================================

    conversation_context = ""

    for msg in request.messages:

        if msg.role == "user":

            conversation_context += " " + msg.content.lower()

    # ==================================================
    # OFF-TOPIC REFUSAL
    # ==================================================

    off_topic_keywords = [
        "legal",
        "salary",
        "labor law",
        "politics",
        "ignore previous instructions"
    ]

    for keyword in off_topic_keywords:

        if keyword in latest_message:

            return {
                "reply": "I can only assist with SHL assessment recommendations and comparisons.",
                "recommendations": [],
                "end_of_conversation": False
            }

    # ==================================================
    # COMPARISON
    # ==================================================

    if "difference between" in latest_message:

        try:

            text = latest_message.replace("difference between", "")

            parts = text.split("and")

            name1 = parts[0].strip()
            name2 = parts[1].strip()

            assessment1 = get_assessment_by_name(name1)
            assessment2 = get_assessment_by_name(name2)

            if assessment1 and assessment2:

                comparison = f"""
{name1.upper()}:
{assessment1.get('description', '')}

{name2.upper()}:
{assessment2.get('description', '')}
"""

                return {
                    "reply": comparison,
                    "recommendations": [],
                    "end_of_conversation": False
                }

        except Exception:

            return {
                "reply": "I could not compare those assessments.",
                "recommendations": [],
                "end_of_conversation": False
            }

    # ==================================================
    # VAGUE QUERY DETECTION
    # ==================================================

    vague_queries = [
        "assessment",
        "test",
        "hiring",
        "need assessment",
        "need test"
    ]

    if len(latest_message.split()) <= 3 or latest_message in vague_queries:

        return {
            "reply": "Could you share more details about the role, seniority level, and skills you want to assess?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # ==================================================
    # RECOMMENDATIONS
    # ==================================================

    results = search_assessments(conversation_context)

    recommendations = []

    for item in results:

        recommendations.append({
            "name": item["name"],
            "url": item["url"],
            "test_type": "Assessment"
        })

    # ==================================================
    # RESPONSE TEXT
    # ==================================================

    reply_text = "Here are recommended SHL assessments."

    if "add" in latest_message:

        reply_text = (
            "I've updated the recommendations "
            "based on your additional requirements."
        )

    # ==================================================
    # FINAL RESPONSE
    # ==================================================

    return {
        "reply": reply_text,
        "recommendations": recommendations,
        "end_of_conversation": True
    }
# ==================================================
# RAILWAY / DEPLOYMENT STARTUP
# ==================================================

if __name__ == "__main__":

    import os
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )
