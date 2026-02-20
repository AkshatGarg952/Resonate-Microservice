"""
Blood report parsing routes.
"""
import asyncio
from fastapi import APIRouter, HTTPException, Depends, Request

from app.models.schemas import ParseRequest
from app.services import pdf_service, openai_service
from app.core.logger import logger, log_request, log_error
from app.core.auth import verify_internal_secret

# Import the shared limiter from main app
# Import the shared limiter from core
from app.core.limiter import limiter

router = APIRouter(dependencies=[Depends(verify_internal_secret)])


@router.post("/parse-report")
@limiter.limit("5/minute")
async def parse_report(request: Request, req: ParseRequest):
    """
    Parse blood report PDF and extract biomarker values.

    - Downloads PDF from URL
    - Classifies if it's a valid blood report
    - Extracts requested biomarker values

    Rate limited to 5 requests/minute — each call costs real OpenAI money.
    """
    log_request("/parse-report")

    # Validate biomarkers list
    if not req.biomarkers:
        raise HTTPException(
            status_code=400,
            detail="biomarkers list cannot be empty."
        )

    # Download PDF (includes file safety guards: 20MB limit, PDF content-type check)
    try:
        pdf_bytes = pdf_service.download_file(req.pdfUrl)
    except Exception as e:
        log_error("PDF download", e)
        raise HTTPException(status_code=400, detail="Could not download PDF")

    loop = asyncio.get_event_loop()

    # Get preview images for classification — run in thread pool (CPU-bound)
    preview_images = await loop.run_in_executor(
        None, lambda: pdf_service.pdf_to_images(pdf_bytes, max_pages=2)
    )
    preview_content = pdf_service.images_to_base64(preview_images)

    # Classify document
    try:
        classification = await openai_service.classify_blood_report(preview_content)
    except Exception as e:
        log_error("Document classification", e)
        raise HTTPException(status_code=500, detail="Failed to classify report type")

    if not classification.get("isBloodReport"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid report type. Reason: {classification.get('reason')}"
        )

    # Convert full PDF for extraction — run in thread pool (CPU-bound)
    full_images = await loop.run_in_executor(
        None, lambda: pdf_service.pdf_to_images(pdf_bytes)
    )
    full_content = pdf_service.images_to_base64(full_images)

    # Extract biomarkers
    try:
        extracted = await openai_service.extract_biomarkers(full_content, req.biomarkers)
    except Exception as e:
        log_error("Biomarker extraction", e)
        raise HTTPException(status_code=500, detail="AI did not return valid JSON")

    # Build response with original biomarker names
    final_values = {}
    for bm in req.biomarkers:
        key = openai_service.sanitize_key(bm)
        value = extracted.get(key)
        final_values[bm] = value if isinstance(value, (int, float)) else None

    missing = [k for k, v in final_values.items() if v is None]

    logger.info(f"Extracted {len(req.biomarkers) - len(missing)}/{len(req.biomarkers)} biomarkers")

    return {
        "confidence": classification.get("confidence"),
        "totalBiomarkers": len(req.biomarkers),
        "foundBiomarkers": len(req.biomarkers) - len(missing),
        "missingBiomarkers": missing,
        "values": final_values
    }
