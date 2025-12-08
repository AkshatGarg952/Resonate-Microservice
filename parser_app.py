import io
import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz
from google.generativeai import configure, GenerativeModel
import json
import re
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Diagnostics AI Parser (Gemini + Status)")

API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=API_KEY)

model = GenerativeModel("gemini-2.5-flash")

class ParseRequest(BaseModel):
    pdfUrl: str

RANGES = {
    "hemoglobin": (12, 16),
    "fastingGlucose": (70, 99),
    "hdl": (40, 200),
    "ldl": (0, 100),
    "triglycerides": (0, 150),
    "tsh": (0.4, 4.0),
    "vitaminD": (30, 100),
    "alt": (7, 56),
    "ast": (10, 40)
}

def calculate_status(name, value):
    if value is None:
        return None
    low, high = RANGES[name]
    if value < low or value > high:
        return "bad"
    return "good"

def download_pdf(url):
    r = requests.get(url)
    r.raise_for_status()
    return r.content

def pdf_to_images(pdf_bytes):
    """Convert PDF bytes to list of image bytes using PyMuPDF"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    img_buffers = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) 
        
        
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_buffers.append(buf.getvalue())
    
    doc.close()
    return img_buffers

@app.get("/")
def root():
    return {"message": "Diagnostics AI Parser is running."}

@app.post("/parse-report")
def parse_report(req: ParseRequest):
    try:
        pdf_bytes = download_pdf(req.pdfUrl)
    except:
        raise HTTPException(400, "Could not download PDF")

    
    img_buffers = pdf_to_images(pdf_bytes)

    prompt = """
    Extract ONLY these biomarkers from the blood report:

hemoglobin, fasting glucose, hdl, ldl, triglycerides, tsh, vitamin d, alt, ast.

If a numeric value is directly mentioned, return that exact value.

If a numeric value is not mentioned, infer an approximate numeric value based on any interpretation text (such as "normal", "slightly elevated", "borderline high", "low", etc.) and standard clinical reference ranges for an adult.

If no numeric value AND no interpretation related to that biomarker is provided, return null for that particular field.

Behave like a medical expert and assign the most probable numeric value based only on the information present in the report.

Return JSON only in the following format:
{
  "hemoglobin": 13.5,
  "fastingGlucose": 98,
  "hdl": 45,
  "ldl": 110,
  "triglycerides": 120,
  "tsh": 2.1,
  "vitaminD": 25,
  "alt": 32,
  "ast": 28
}
    """

    response = model.generate_content(
        [
            prompt,
            *[{"mime_type": "image/jpeg", "data": img} for img in img_buffers]
        ]
    )

    raw = response.text
    try:
        text_json = re.search(r"\{.*\}", raw, re.DOTALL).group(0)
        values = json.loads(text_json)
    except:
        raise HTTPException(500, "Gemini did not return valid JSON")

    final = {}
    for key, value in values.items():
        if value is None:
            final[key] = {"value": None, "status": None}
        else:
            final[key] = {
                "value": value,
                "status": calculate_status(key, float(value))
            }

    return final