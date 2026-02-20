"""
OpenAI API service for AI operations.
"""
import json
import re
import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from app.core.config import settings
from app.core.logger import logger, log_ai_call, log_error


# Initialize OpenAI client
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Per-call timeout: 10s to connect, 90s to receive response.
# Prevents a stalled OpenAI stream from hanging a uvicorn worker forever.
# The tenacity retry decorator will fire a new attempt if the timeout raises.
OPENAI_TIMEOUT = openai.Timeout(total=90.0, connect=10.0, read=90.0, write=10.0)

# Tenacity retry policy: 3 total attempts, exponential backoff 2s→10s
# Only retries transient errors: rate limits and connection failures
_openai_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError)),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)


@_openai_retry
async def call_vision_api(
    prompt: str,
    image_content: list[dict],
    temperature: float = 0.0
) -> dict:
    """
    Call OpenAI Vision API with images.

    Retries automatically on RateLimitError / APIConnectionError
    (up to 3 attempts with exponential backoff).

    Args:
        prompt: Text prompt for the model
        image_content: List of image content dicts
        temperature: Model temperature

    Returns:
        Parsed JSON response

    Raises:
        ValueError: If response is not valid JSON
        openai.RateLimitError: If all retries exhausted
    """
    log_ai_call("Vision API", settings.OPENAI_MODEL)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            *image_content
        ]
    }]

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        timeout=OPENAI_TIMEOUT,  # Hard cap: stalled responses won't hang the worker
    )

    try:
        result = json.loads(response.choices[0].message.content)
        logger.info("Vision API call successful")
        return result
    except json.JSONDecodeError as e:
        log_error("Vision API JSON parsing", e)
        raise ValueError("AI did not return valid JSON")


@_openai_retry
async def call_chat_api(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.7
) -> dict:
    """
    Call OpenAI Chat API for text generation.

    Retries automatically on RateLimitError / APIConnectionError
    (up to 3 attempts with exponential backoff).

    Args:
        system_prompt: System context
        user_prompt: User request
        temperature: Model temperature

    Returns:
        Parsed JSON response

    Raises:
        ValueError: If response is not valid JSON
        openai.RateLimitError: If all retries exhausted
    """
    log_ai_call("Chat API", settings.OPENAI_MODEL)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
        timeout=OPENAI_TIMEOUT,  # Hard cap: stalled responses won't hang the worker
    )

    try:
        result = json.loads(response.choices[0].message.content)
        logger.info("Chat API call successful")
        return result
    except json.JSONDecodeError as e:
        log_error("Chat API JSON parsing", e)
        raise ValueError("AI did not return valid JSON")


def sanitize_key(name: str) -> str:
    """
    Convert biomarker name to valid JSON key.
    
    Args:
        name: Biomarker display name
        
    Returns:
        camelCase key string
    """
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', name.lower())
    words = cleaned.split()
    if not words:
        return "biomarker"
    return words[0] + ''.join(w.capitalize() for w in words[1:])


# --- Blood Report Classification ---

CLASSIFICATION_PROMPT = """
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


async def classify_blood_report(image_content: list[dict]) -> dict:
    """
    Classify if document is a blood report.
    
    Args:
        image_content: Document page images
        
    Returns:
        Classification result with confidence
    """
    return await call_vision_api(
        CLASSIFICATION_PROMPT,
        image_content,
        temperature=settings.TEMPERATURE_EXTRACTION
    )


async def extract_biomarkers(image_content: list[dict], biomarkers: list[str]) -> dict:
    """
    Extract biomarker values from blood report images.
    
    Args:
        image_content: Report page images
        biomarkers: List of biomarker names to extract
        
    Returns:
        Dict mapping biomarker names to values
    """
    biomarker_list = "\n".join([f"- {bm}" for bm in biomarkers])
    json_schema = {sanitize_key(bm): None for bm in biomarkers}

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

    return await call_vision_api(
        prompt,
        image_content,
        temperature=settings.TEMPERATURE_EXTRACTION
    )


# --- Workout Generation ---

WORKOUT_SYSTEM_PROMPT = """
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
2. Adapt volume/intensity based on Age and Cycle Phase (e.g., Luteal = lower intensity/steady state; Follicular = HIIT/Strength).
3. Use ONLY available equipment.
4. FACTOR IN MOTIVATION: 
   - Low Motivation: Focus on "easy wins", shorter sets, fun exercises, lower barrier to entry.
   - High Motivation: Push limits, high intensity, complex movements.
5. FACTOR IN TIMING:
   - Morning: Energizing, mobility-focused.
   - Evening: De-stressing, avoid over-stimulation if late, focus on recovery/strength.
6. ADDRESS BARRIERS:
   - Time constraints: Supersets, minimal rest.
   - Boredom: High variety, novel exercises.
   - Low Energy: Start slow, build momentum.
"""


async def generate_workout(
    level: str,
    equipment: list[str],
    time: int,
    injuries: list[str],
    motivation: str = None,
    timing: str = None,
    barriers: list[str] = None,
    age: int = None,
    gender: str = None,
    weight: float = None,
    cycle_phase: str = None
) -> dict:
    """
    Generate personalized workout plan.
    
    Returns:
        Workout plan dict
    """
    profile_desc = f"Fitness Level: {level}\nTime Available: {time} minutes\n"
    
    if equipment:
        profile_desc += f"Equipment: {', '.join(equipment)}\n"
    else:
        profile_desc += "Equipment: None (Bodyweight only)\n"
        
    if injuries:
        profile_desc += f"Injuries/Limitations: {', '.join(injuries)}\n"
        
    if motivation:
        profile_desc += f"Motivation Level: {motivation}\n"
    if timing:
        profile_desc += f"Preferred Workout Time: {timing}\n"
    if barriers:
        profile_desc += f"Barriers/Challenges: {', '.join(barriers)}\n"
    if age:
        profile_desc += f"Age: {age}\n"
    if gender:
        profile_desc += f"Gender: {gender}\n"
    if weight:
        profile_desc += f"Weight: {weight}kg\n"
    if cycle_phase and gender and gender.lower() == 'female':
        profile_desc += f"Menstrual Cycle Phase: {cycle_phase}\n"

    user_prompt = f"Create a workout for this user:\n{profile_desc}"
    
    return await call_chat_api(
        WORKOUT_SYSTEM_PROMPT,
        user_prompt,
        temperature=settings.TEMPERATURE_CREATIVE
    )


# --- Nutrition Generation ---

async def generate_meal_plan(
    age: int = None,
    gender: str = None,
    weight: float = None,
    height: float = None,
    goals: str = None,
    diet_type: str = None,
    allergies: list[str] = None,
    cuisine: str = "Indian"
) -> dict:
    """
    Generate daily meal plan.
    
    Returns:
        Meal plan dict
    """
    allergies_str = ", ".join(allergies) if allergies else "None"
    
    prompt = f"""
You are an expert nutritionist specializing in {cuisine} cuisine.
Create a daily meal plan for a user with the following profile:

- Age: {age}
- Gender: {gender}
- Weight: {weight}kg
- Height: {height}cm
- Goals: {goals}
- Diet Type: {diet_type}
- Allergies/Restrictions: {allergies_str}

Provide a structured meal plan with:
1. Breakfast
2. Lunch
3. Dinner
4. Snacks (2 options)

Focus on healthy, nutritious {cuisine} meals that align with the user's goals.
Include approximate calories and protein for each meal.

Return JSON ONLY with this structure:
{{
  "breakfast": {{ "name": "...", "description": "...", "calories": 0, "protein": "0g" }},
  "lunch": {{ "name": "...", "description": "...", "calories": 0, "protein": "0g" }},
  "dinner": {{ "name": "...", "description": "...", "calories": 0, "protein": "0g" }},
  "snacks": [
    {{ "name": "...", "description": "...", "calories": 0, "protein": "0g" }}
  ],
  "total_calories": 0,
  "total_protein": "0g"
}}
"""

    return await call_chat_api(
        "You are a helpful nutrition assistant that outputs strictly valid JSON.",
        prompt,
        temperature=settings.TEMPERATURE_CREATIVE
    )


# --- Food Analysis ---

async def analyze_food_image(image_base64: str, cuisine: str = "General") -> dict:
    """
    Analyze food image for nutritional content.
    
    Args:
        image_base64: Base64 encoded image
        cuisine: Cuisine context hint
        
    Returns:
        Food analysis dict
    """
    prompt = f"""
You are an expert nutritionist and food analyst. 
Analyze the food in this image. The user specified the cuisine/context as: {cuisine}.

Identify the food items, estimate portion sizes, and calculate nutritional values.

Return JSON ONLY with this structure:
{{
  "food_name": "Name of the dish/food",
  "description": "Brief description",
  "ingredients": ["List of likely ingredients"],
  "nutritional_info": {{
      "calories": 0,
      "protein": "0g",
      "carbohydrates": "0g",
      "fats": "0g",
      "fiber": "0g"
  }},
  "health_rating": "A score from 1-10 (10 being very healthy)",
  "suggestions": "Suggestions to make it healthier or what to pair it with"
}}
"""

    image_content = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
    }]

    return await call_vision_api(
        prompt,
        image_content,
        temperature=settings.TEMPERATURE_ANALYSIS
    )
