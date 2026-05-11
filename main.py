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

    latest_message = request.messages[-1].content.lower().strip()

    # ==================================================
    # BUILD CONVERSATION CONTEXT
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
        "ignore previous instructions",
        "religion",
        "medical advice"
    ]

    for keyword in off_topic_keywords:

        if keyword in latest_message:

            return {
                "reply": (
                    "I can only assist with SHL assessment "
                    "recommendations and comparisons."
                ),
                "recommendations": [],
                "end_of_conversation": False
            }

    # ==================================================
    # ASSESSMENT COMPARISON
    # ==================================================

    if "difference between" in latest_message:

        try:

            text = latest_message.replace(
                "difference between",
                ""
            )

            parts = text.split("and")

            if len(parts) >= 2:

                name1 = parts[0].strip()
                name2 = parts[1].strip()

                assessment1 = get_assessment_by_name(name1)
                assessment2 = get_assessment_by_name(name2)

                if assessment1 and assessment2:

                    comparison = f"""
{name1.upper()}:
{assessment1.get('description', 'No description available.')}

{name2.upper()}:
{assessment2.get('description', 'No description available.')}
"""

                    return {
                        "reply": comparison,
                        "recommendations": [],
                        "end_of_conversation": False
                    }

            return {
                "reply": (
                    "Please provide two valid SHL "
                    "assessments to compare."
                ),
                "recommendations": [],
                "end_of_conversation": False
            }

        except Exception:

            return {
                "reply": (
                    "I could not compare those "
                    "assessments."
                ),
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
        "need test",
        "recommend assessment"
    ]

    if (
        len(latest_message.split()) <= 3
        or latest_message in vague_queries
    ):

        return {
            "reply": (
                "Could you share more details such as "
                "the role, seniority level, technical "
                "skills, and whether you need "
                "technical, cognitive, or personality "
                "assessments?"
            ),
            "recommendations": [],
            "end_of_conversation": False
        }

    # ==================================================
    # RECOMMENDATION SEARCH
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
    # REFINEMENT DETECTION
    # ==================================================

    refinement_keywords = [
        "add",
        "also",
        "include",
        "plus",
        "need"
    ]

    is_refinement = False

    for word in refinement_keywords:

        if word in latest_message:

            is_refinement = True

    # ==================================================
    # RESPONSE TEXT
    # ==================================================

    if is_refinement:

        reply_text = (
            "I've updated the recommendations "
            "based on your additional requirements."
        )

    else:

        reply_text = (
            "Here are recommended SHL assessments."
        )

    # ==================================================
    # CONVERSATION END DETECTION
    # ==================================================

    done_keywords = [
        "thanks",
        "thank you",
        "done",
        "looks good",
        "perfect"
    ]

    conversation_done = False

    for word in done_keywords:

        if word in latest_message:

            conversation_done = True

    # ==================================================
    # FINAL RESPONSE
    # ==================================================

    return {
        "reply": reply_text,
        "recommendations": recommendations,
        "end_of_conversation": conversation_done
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
