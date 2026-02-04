from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

# Create the FastAPI app instance
app = FastAPI()

# Create OpenAI client using the API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Workflow ID (Ask the TDA agent workflow)
WORKFLOW_ID = os.getenv("WORKFLOW_ID")


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_tda(q: Question):
    """
    Receive a question from the client,
    send it to the Ask the TDA workflow,
    and return the model's answer.
    """
    response = client.responses.create(
        model="gpt-4.1",
        input=q.question,
        workflow=WORKFLOW_ID,
    )

    # Basic way to get the text output from the response object
    # This may need tweaking depending on exact response schema,
    # but it won't stop the server from starting.
    try:
        answer_text = response.output[0].content[0].text
    except Exception:
        # Fallback: just string the whole response if schema changes
        answer_text = str(response)

    return {
        "answer": answer_text
    }
