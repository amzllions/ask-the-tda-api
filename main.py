from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

# Create FastAPI app
app = FastAPI()

# OpenAI client (API key comes from Render environment variables)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Workflow ID for "Ask the TDA" agent
WORKFLOW_ID = os.getenv("WORKFLOW_ID")


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_tda(q: Question):
    try:
        # Call the Ask the TDA agent via Responses API
        response = client.responses.create(
            model="gpt-4.1",
            input=q.question,
            metadata={
                "workflow_id": WORKFLOW_ID
            }
        )

        # Extract text output safely
        answer_text = ""

        if response.output:
            for item in response.output:
                if "content" in item:
                    for block in item["content"]:
                        if block.get("type") == "output_text":
                            answer_text += block.get("text", "")

        if not answer_text:
            answer_text = "No answer returned from the TDA agent."

        return {
            "
