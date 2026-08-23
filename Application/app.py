from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize our FastAPI App
app = FastAPI(title="Text Summarizer App", description="Text SUmmarization using T5", version="1.0")

# Model & Tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model") # To use the saved model
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Templating
templates = Jinja2Templates(directory=".") # HTML, CSS Files are available in the same folder

# Input Schema for Dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # Removing lines
    text = re.sub(r"\s+", " ", text) # Removing extra spaces
    text = re.sub(r"<.*?>", " ", text) # Removing html tags
    text = text.strip().lower() # Removing trailing spaces and converting to lowercase
    return text

def summarize_dialogue(dialogue: str) -> str: # -> str means returning string value
    # Clean the dialogue
    dialogue = clean_data(dialogue)

    # Tokenize
    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt" # pytorch tensors
    ).to(device)

    # Generating Summary => We will get Token ids
    model.to(device)
    generated_ids = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4, # Generates 4 diffierent outputs & chooses the best out of 4
        early_stopping = True
    )

    # Decoding Token ids to get the text summary
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary

# API End Points
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}

@app.get("/", response_class=HTMLResponse) # Home Page
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Run this file by typing in the vs code terminal below: uvicorn app:app --reload
# To increase the performance of the model: We can use all the 14,000 samples of train_data instead of 4000.
# We can use 100 epochs instead of 6, but it will be very time consuming
# We can use t5-Base, t5-Large, or others..., but currently we have used t5-small
