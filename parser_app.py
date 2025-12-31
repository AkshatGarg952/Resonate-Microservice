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

app = FastAPI(title="Diagnostics AI Parser (Extraction Only)")

API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=API_KEY)

model = GenerativeModel("gemini-2.5-flash")

# -------------------------
# Request Model
# -------------------------
class ParseRequest(BaseModel):
    pdfUrl: str
    biomarkers: list[str]  # List of biomarker names to extract


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
    
    # Step 3: Validate biomarkers list
    if not req.biomarkers or len(req.biomarkers) == 0:
        raise HTTPException(
            status_code=400,
            detail="biomarkers list cannot be empty. Please provide at least one biomarker name."
        )
    
    
    full_images = pdf_to_images(pdf_bytes)
     
    # Step 5: Generate dynamic extraction prompt based on provided biomarkers
    # Create a list of biomarker names for the prompt
    biomarker_list = ",\n".join([f"- {bm}" for bm in req.biomarkers])
    
    # Create JSON format example with camelCase keys (sanitized biomarker names)
    def sanitize_key(name: str) -> str:
        """Convert biomarker name to camelCase JSON key"""
        # Remove special characters, convert to lowercase, replace spaces with camelCase
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
        words = cleaned.split()
        if not words:
            return "biomarker"
        return words[0] + ''.join(word.capitalize() for word in words[1:])
    
    json_format_lines = []
    for bm in req.biomarkers:
        key = sanitize_key(bm)
        json_format_lines.append(f'  "{key}": number | null')
    
    json_format = "{\n" + ",\n".join(json_format_lines) + "\n}"
    
    prompt = f"""
Extract ONLY these biomarkers from the blood report:

{biomarker_list}

STRICT RULES:
- Extract the EXACT numeric value ONLY if explicitly written in the report.
- If a biomarker is mentioned WITHOUT a numeric value, return null for that biomarker.
- If a biomarker is NOT mentioned at all in the report, return null for that biomarker.
- Do NOT infer, estimate, or guess values.
- Do NOT apply medical knowledge.
- Do NOT calculate ranges or status.
- Match biomarker names flexibly (case-insensitive, handle abbreviations and variations).

Return JSON ONLY in this format:
{json_format}

IMPORTANT: Return ALL biomarkers in the response, even if their value is null.
"""

    response = model.generate_content(
        [prompt, *[{"mime_type": "image/jpeg", "data": img} for img in full_images]]
    )

    # Step 6: Parse AI output
    try:
        text_json = re.search(r"\{.*\}", response.text, re.DOTALL).group(0)
        extracted_values = json.loads(text_json)
    except Exception:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON")

    
    # Create a mapping of sanitized keys to original biomarker names
    biomarker_key_map = {sanitize_key(bm): bm for bm in req.biomarkers}
    
    # Build final values dict ensuring all biomarkers are included
    final_values = {}
    for bm in req.biomarkers:
        key = sanitize_key(bm)
        # Check if the key exists in extracted values (try both the sanitized key and original name)
        value = extracted_values.get(key) or extracted_values.get(bm.lower()) or extracted_values.get(bm)
        final_values[bm] = value if value is not None and isinstance(value, (int, float)) else None

    
    missing_biomarkers = [bm for bm, v in final_values.items() if v is None]

    
    return {
        "confidence": classification.get("confidence"),
        "totalBiomarkers": len(req.biomarkers),
        "foundBiomarkers": len(req.biomarkers) - len(missing_biomarkers),
        "missingBiomarkers": missing_biomarkers,
        "values": final_values
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
