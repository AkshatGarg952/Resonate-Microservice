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

# -------------------------
# App Setup
# -------------------------
app = FastAPI(title="Diagnostics AI Parser (Extraction Only)")

API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=API_KEY)

model = GenerativeModel("gemini-2.5-flash")

# -------------------------
# Request Model
# -------------------------
class ParseRequest(BaseModel):
    pdfUrl: str


# -------------------------
# Utilities
# -------------------------
def download_pdf(url: str) -> bytes:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content


def pdf_to_images(pdf_bytes: bytes, max_pages: int | None = None):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)

    for page_num in range(pages_to_process):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        images.append(buf.getvalue())

    doc.close()
    return images


def classify_report_is_blood(img_buffers: list[bytes]) -> dict:
    prompt = """
You are a medical document classifier.

Based ONLY on the document content,
determine whether this is a BLOOD TEST REPORT.

Return JSON only:
{
  "isBloodReport": true | false,
  "confidence": "low" | "medium" | "high",
  "reason": "short explanation"
}
"""

    response = model.generate_content(
        [prompt, *[{"mime_type": "image/jpeg", "data": img} for img in img_buffers]]
    )

    try:
        text_json = re.search(r"\{.*\}", response.text, re.DOTALL).group(0)
        return json.loads(text_json)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify report type")


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Diagnostics AI Parser (Extraction Only) running"}


@app.post("/parse-report")
def parse_report(req: ParseRequest):
    """
    IMPORTANT:
    - This endpoint ONLY extracts raw numeric values
    - NO medical logic
    - NO reference ranges
    - NO good/bad status
    """

    # Step 1: Download PDF
    try:
        pdf_bytes = download_pdf(req.pdfUrl)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not download PDF")

    # Step 2: Preview classification
    preview_images = pdf_to_images(pdf_bytes, max_pages=2)
    classification = classify_report_is_blood(preview_images)

    if not classification.get("isBloodReport"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type. Reason: {classification.get('reason')}"
        )

    # Step 3: Convert full PDF to images
    full_images = pdf_to_images(pdf_bytes)

    # Step 4: STRICT extraction prompt
    prompt = """
Extract ONLY these biomarkers from the blood report:

hemoglobin,
fasting glucose,
hdl,
ldl,
triglycerides,
tsh,
vitamin d,
alt,
ast

STRICT RULES:
- Extract the EXACT numeric value ONLY if explicitly written.
- If mentioned WITHOUT a numeric value, return null.
- If NOT mentioned at all, return null.
- Do NOT infer, estimate, or guess.
- Do NOT apply medical knowledge.
- Do NOT calculate ranges or status.

Return JSON ONLY in this format:
{
  "hemoglobin": number | null,
  "fastingGlucose": number | null,
  "hdl": number | null,
  "ldl": number | null,
  "triglycerides": number | null,
  "tsh": number | null,
  "vitaminD": number | null,
  "alt": number | null,
  "ast": number | null
}
"""

    response = model.generate_content(
        [prompt, *[{"mime_type": "image/jpeg", "data": img} for img in full_images]]
    )

    # Step 5: Parse AI output
    try:
        text_json = re.search(r"\{.*\}", response.text, re.DOTALL).group(0)
        values = json.loads(text_json)
    except Exception:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON")

    # Step 6: Identify missing biomarkers
    missing_biomarkers = [k for k, v in values.items() if v is None]

    # Step 7: RETURN EXTRACTION ONLY
    return {
        "confidence": classification.get("confidence"),
        "missingBiomarkers": missing_biomarkers,
        "values": values
    }


# -------------------------
# Local Run
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "parser_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
