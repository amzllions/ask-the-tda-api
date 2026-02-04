from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WORKFLOW_ID = os.getenv("WORKFLOW_ID")


class Question(BaseModel):
    question: str


SYSTEM_INSTRUCTIONS = """
You are “Ask the TDA”, a rules assistant for poker tournament officials.

CRITICAL BEHAVIOUR RULES (OVERRIDE EVERYTHING ELSE):

1. You may ONLY use information contained in the official TDA (Tournament Directors Association) rules documents and TDA summit materials you were configured with.
2. You MUST NOT use Robert’s Rules of Poker, WSOP rules, general “casino rules”, online articles, or any other external sources or prior poker knowledge.
3. If the situation is not explicitly covered by the TDA rules, you MUST clearly say that the TDA rules do not provide a specific ruling and that the final decision is at the discretion of the Tournament Director.
4. Every answer MUST follow this exact format and keep the language of the user’s question:

The answer in this situation is:
[short, clear ruling in the same language as the question]

Relevant TDA Rule(s):
- Rule X
- Rule Y

5. Do NOT invent rule numbers, procedures, or penalties.
6. Do NOT mention or reference Robert’s Rules, WSOP, “most casinos”, or any other non-TDA authority.
"""


@app.post("/ask")
async def ask_tda(q: Question):
    try:
        # Build a strict prompt: instructions + user question
        prompt = SYSTEM_INSTRUCTIONS + "\n\nQuestion:\n" + q.question.strip()

        # Call the Ask the TDA workflow
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
            metadata={"workflow_id": WORKFLOW_ID},
        )

        # Extract text correctly from Responses API
        answer_text = ""
        if hasattr(response, "output") and response.output:
            for message in response.output:
                for content in message.content:
                    if content.type == "output_text":
                        # .text is already a string
                        answer_text += content.text

        # Backend guardrails: block obviously bad sources
        lower = answer_text.lower()

        banned_terms = [
            "robert's rules",
            "roberts rules",
            "wsop",
            "world series of poker",
            "pokernews.com",
            "pokercoach",
            "roberts-rules-of-poker",
        ]

        if any(term in lower for term in banned_terms):
            # Hard override to a safe, honest fallback
            answer_text = (
                "The answer in this situation is:\n"
                "The TDA rules do not explicitly cover this exact situation. "
                "The Tournament Director should apply TDA Rule 1 and make the decision "
                "that best maintains fairness and the integrity of the game.\n\n"
                "Relevant TDA Rule(s):\n"
                "- Rule 1 (TD discretion)"
            )

        # Make sure there is at least a minimal TDA-style footer
        if "Relevant TDA Rule" not in answer_text:
            answer_text += (
                "\n\nRelevant TDA Rule(s):\n"
                "- Rule 1 (TD discretion)"
            )

        return {"answer": answer_text}

    except Exception as e:
        # Never hang: always return something JSON-valid
        return {"error": str(e)}
