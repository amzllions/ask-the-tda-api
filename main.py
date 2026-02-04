from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os
from functools import lru_cache

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@lru_cache
def load_tda_rules() -> str:
    """
    Load the full TDA rules text from a local file.
    This is your single source of truth.
    """
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "tda_rules_en.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_tda(q: Question):
    try:
        tda_rules = load_tda_rules()

        # Build a strict prompt: model only sees TDA rules + the question
        prompt = f"""
You are a strict rules assistant for poker tournament directors.

Below is the full text of the official Poker TDA rules.
You must obey ALL of these constraints:

- You may ONLY use the rules text below.
- You must NOT use any other poker rule set or general poker knowledge
  (no Robert's Rules, no WSOP rules, no "standard casino procedure", etc.).
- If the rules do not explicitly cover the situation, say that clearly and
  refer to TDA Rule 1 (TD discretion).
- You must answer in the SAME LANGUAGE as the user's question.
- You must include the exact TDA rule number(s) that support your decision.
- Use EXACTLY this format:

The answer in this situation is:
[clear ruling in same language as the question]

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

        # Extract text from Responses API
        answer_text = ""
        if hasattr(response, "output") and response.output:
            for message in response.output:
                for content in message.content:
                    if content.type == "output_text":
                        answer_text += content.text

        answer_text = answer_text.strip()

        # If nothing useful came back
        if not answer_text:
            answer_text = (
                "The TDA rules text did not provide a clear explicit answer to this situation. "
                "The Tournament Director should apply TDA Rule 1 and make the fairest possible ruling.\n\n"
                "Relevant TDA Rule(s):\n"
                "- Rule 1 (TD discretion)"
            )

        # Backend guardrail: block non-TDA sources if they somehow appear
        lower = answer_text.lower()
        banned_terms = [
            "robert's rules",
            "roberts rules",
            "wsop",
            "world series of poker",
            "pokernews",
            "roberts-rules-of-poker",
        ]
        if any(term in lower for term in banned_terms):
            answer_text = (
                "The TDA rules assistant must not rely on non-TDA sources. "
                "For this situation, please consult the official TDA rules text directly "
                "and apply TDA Rule 1 if necessary.\n\n"
                "Relevant TDA Rule(s):\n"
                "- Rule 1 (TD discretion)"
            )

        # Ensure we always end with a TDA-style footer
        if "relevant tda rule" not in answer_text.lower():
            answer_text += (
                "\n\nRelevant TDA Rule(s):\n"
                "- Rule 1 (TD discretion)"
            )

        return {"answer": answer_text}

    except Exception as e:
        return {"error": str(e)}
