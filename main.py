from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WORKFLOW_ID = os.getenv("WORKFLOW_ID")


class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_tda(q: Question):
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=q.question,
            metadata={"workflow_id": WORKFLOW_ID},
        )

        answer_text = ""

        if hasattr(response, "output") and response.output:
            for item in response.output:
                content = item.get("content", [])
                for block in content:
                    if block.get("type") == "output_text":
                        answer_text += block.get("text", "")

        if not answer_text:
            answer_text = "No answer returned from the TDA agent."

        return {"answer": answer_text}

    except Exception as e:
        return {"error": str(e)}
