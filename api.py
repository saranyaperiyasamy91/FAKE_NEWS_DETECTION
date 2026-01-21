from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load trained model
tokenizer = DistilBertTokenizerFast.from_pretrained("fake_news_model")
model = DistilBertForSequenceClassification.from_pretrained("fake_news_model")
model.eval()

# UI page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# Prediction
@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, news_text: str = Form(...)):
    inputs = tokenizer(
        news_text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
         outputs = model(**inputs)
         logits = outputs.logits

         probs = torch.softmax(logits, dim=1)
         confidence = torch.max(probs).item()
         pred = torch.argmax(probs, dim=1).item()

    if pred == 1:
        result = "🟢 REAL NEWS"
        color = "real"
    else:
        result = "🔴 FAKE NEWS"
        color = "fake"
        
    confidence_text = f"Confidence: {confidence * 100:.2f}%"
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "color": color,
            "confidence": confidence_text
        }
    )
