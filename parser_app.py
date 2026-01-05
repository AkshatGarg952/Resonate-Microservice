import io
import os
import re
import json
import base64
import requests
import fitz
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()


app = FastAPI(title="Diagnostics AI Parser (Extraction Only)")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4.1-mini" 


class ParseRequest(BaseModel):
    pdfUrl: str
    biomarkers: list[str]

class WorkoutRequest(BaseModel):
    fitnessLevel: str
    equipment: list[str]
    timeAvailable: int
    injuries: list[str]
    age: int | None = None
    gender: str | None = None
    weight: float | None = None
    cyclePhase: str | None = None


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


def images_to_openai_content(images: list[bytes]):
    content = []
    for img in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
                }
            }
        )
    return content



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

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *images_to_openai_content(img_buffers)
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to classify report type")



@app.get("/")
def root():
    return {"message": "Diagnostics AI Parser (Extraction Only) running"}


@app.post("/parse-report")
def parse_report(req: ParseRequest):

    if not req.biomarkers:
        raise HTTPException(
            status_code=400,
            detail="biomarkers list cannot be empty."
        )

    try:
        pdf_bytes = download_pdf(req.pdfUrl)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not download PDF")

    preview_images = pdf_to_images(pdf_bytes, max_pages=2)
    classification = classify_report_is_blood(preview_images)

    if not classification.get("isBloodReport"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type. Reason: {classification.get('reason')}"
        )

    full_images = pdf_to_images(pdf_bytes)

    
    biomarker_list = "\n".join([f"- {bm}" for bm in req.biomarkers])

    def sanitize_key(name: str) -> str:
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
        words = cleaned.split()
        return words[0] + ''.join(w.capitalize() for w in words[1:]) if words else "biomarker"

    json_schema = {
        sanitize_key(bm): None for bm in req.biomarkers
    }

    prompt = f"""
Extract ONLY these biomarkers from the blood report:

{biomarker_list}

STRICT RULES:
- Extract the EXACT numeric value ONLY if explicitly written.
- If missing or unclear, return null.
- Do NOT infer or calculate.
- Do NOT apply medical knowledge.
- Match biomarker names flexibly.

Return JSON ONLY matching this schema:
{json.dumps(json_schema, indent=2)}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *images_to_openai_content(full_images)
            ]
        }
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        extracted = json.loads(response.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON")

    
    final_values = {}
    for bm in req.biomarkers:
        key = sanitize_key(bm)
        value = extracted.get(key)
        final_values[bm] = value if isinstance(value, (int, float)) else None

    missing = [k for k, v in final_values.items() if v is None]

    return {
        "confidence": classification.get("confidence"),
        "totalBiomarkers": len(req.biomarkers),
        "foundBiomarkers": len(req.biomarkers) - len(missing),
        "missingBiomarkers": missing,
        "values": final_values
    }


def generate_ai_plan(level: str, equipment: list[str], time: int, injuries: list[str], age: int=None, gender: str=None, weight: float=None, cyclePhase: str=None):
    
    # Construct User Profile String
    profile_desc = f"Fitness Level: {level}\nTime Available: {time} minutes\n"
    if equipment:
        profile_desc += f"Equipment: {', '.join(equipment)}\n"
    else:
        profile_desc += "Equipment: None (Bodyweight only)\n"
        
    if injuries:
        profile_desc += f"Injuries/Limitations: {', '.join(injuries)}\n"
        
    if age: profile_desc += f"Age: {age}\n"
    if gender: profile_desc += f"Gender: {gender}\n"
    if weight: profile_desc += f"Weight: {weight}kg\n"
    if cyclePhase and gender and gender.lower() == 'female':
        profile_desc += f"Menstrual Cycle Phase: {cyclePhase}\n"

    system_prompt = """
    You are an expert elite fitness coach. Create a highly personalized, semi-structured workout plan.
    
    Output JSON ONLY with this structure:
    {
      "title": "Creative Workout Name",
      "duration": "X Minutes",
      "focus": "Target Area or Goal",
      "warmup": [{"name": "Exercise", "duration": "Time/Reps"}],
      "exercises": [{"name": "Exercise", "sets": Number, "reps": "Range or Time", "notes": "Optional tip"}],
      "cooldown": [{"name": "Exercise", "duration": "Time"}]
    }
    
    RULES:
    1. STRICTLY respect injuries. Do NOT include exercises that aggravate listed injuries.
    2. adapting volume/intensity based on Age and Cycle Phase (e.g., Luteal = lower intensity/steady state; Follicular = HIIT/Strength).
    3. Use ONLY available equipment.
    """

    user_prompt = f"""
    Create a workout for this user:
    {profile_desc}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        raise HTTPException(status_code=500, detail="AI generation failed to produce valid JSON")


@app.post("/generate-workout")
def generate_workout(req: WorkoutRequest):
    try:
        plan = generate_ai_plan(
            req.fitnessLevel,
            req.equipment,
            req.timeAvailable,
            req.injuries,
            req.age,
            req.gender,
            req.weight,
            req.cyclePhase
        )
        return {"status": "success", "plan": plan}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "parser_app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
