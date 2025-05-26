import uvicorn
from fastapi import FastAPI
from tensorflow.keras.models import load_model

from src.llm import LLM
from src.model_router import model_router

app = FastAPI()

model = load_model('src/model/model_en22.h5')
with open('src/model/class_labels.txt', 'r') as f:
 classes = [line.strip() for line in f]

app.state.model = model
app.state.classes = classes

llm = LLM("3uTKE448T558Qmem6pBSbvW54nHBR4FP6Xnn6jCl")
app.state.llm = llm
@app.get("/")
async def health_check():
    return {"status": "OK"}

app.include_router(model_router, prefix="/model", tags=["model"])

if __name__ == '__main__':
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)