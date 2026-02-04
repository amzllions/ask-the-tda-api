from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from functools import lru_cache

app = FastAPI()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache
def load_tda_rules() -> str:
    """
    Load the full official TDA rules text.
    This is the ONLY knowledge source the model is allowed to use.
    """
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "tda-rules-en.txt")  # <-- hyphens, no underscores

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_tda(q: Question):
    try:
        tda_rules = load_tda_rules()

        prompt = f"""
You are Ask the TDA, a STRICT poker tournament rules assistant.

You MUST follow these rules without exception:

1. You may ONLY use the official Poker TDA rules text provided below.
2. You must NOT use any other poker rules or general poker knowledge
   (NO Robert’s Rules, NO WSOP rules, NO casino practices).
3. If the TDA rules do NOT explicitly cover the situation, you MUST say so
   and refer to TDA Rule 1 (Tournament Director discretion).
4. You must answer in the SAME LANGUAGE as the question.
5. You must cite the EXACT TDA rule number(s) that support the ruling.
6. You must use EXACTLY this format:

The answer in this situation is:
[clear ruling]

Relevant TDA Rule(s):
- Rule X
- Rule Y

--- START OF OFFICIAL TDA RULES ---
{tda_rules}
--- END OF OFFICIAL TDA RULES ---

User question:
{q.question}
"""

        response = client.responses.create(
            model="gpt-4.1",
            input=prompt,
        )

        # Extract text from the Responses API
        answer_text = ""
        if response.output:
            for message in response.output:
                for content in message.content:
                    if content.type == "output_text":
                        answer_text += content.text

        answer_text = answer_text.strip()

        # Safety fallback if model returns nothing useful
        if not answer_text:
            answer_text = (
                "The answer in this situation is:\n"
                "The TDA rules do not explicitly cover this situation. "
                "The Tournament Director should make a ruling that best "
                "preserves fairness and the integrity of the tournament.\n\n"
                "Relevant TDA Rule(s):\n"
                "- Rule 1"
            )

        # HARD BLOCK non-TDA sources (extra safety)
        banned_terms = [
            "robert",
            "wsop",
            "world series of poker",
            "casino rules",
            "standard procedure",
        ]

        if any(term in answer_text.lower() for term in banned_terms):
            answer_text = (
                "The answer in this situation is:\n"
                "The TDA rules do not explicitly cover this situation. "
                "The Tournament Director should make the final ruling "
                "under TDA Rule 1.\n\n"
                "Relevant TDA Rule(s):\n"
                "- Rule 1"
            )

        # Ensure rule section is always present
        if "Relevant TDA Rule" not in answer_text:
            answer_text += "\n\nRelevant TDA Rule(s):\n- Rule 1"

        return {"answer": answer_text}

    except Exception as e:
        return {"error": str(e)}
