from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Dict
from rag import Rag

app = FastAPI()

# Initialize the Rag object once
rag_object = Rag(url="https://www.w3schools.com/html/html_tables.asp")

class Message(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    message: List[Message]


@app.post("/response/")
async def get_response(messages: ChatResponse):
    try:
       
        message_dict = {"message": [{"role": msg.role, "content": msg.content} for msg in messages.message]}
        response = rag_object.response(message_dict)
        return response['answer']
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 