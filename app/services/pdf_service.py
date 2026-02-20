"""
PDF processing service - download, convert to images.
"""
import io
import base64
import requests
from PIL import Image
import fitz  # PyMuPDF

from app.core.config import settings
from app.core.logger import logger, log_error


def download_file(url: str) -> bytes:
    """
    Download a file from URL with safety guards.

    Rejects files > 20MB or non-PDF content types to prevent
    malicious URLs from crashing the process.

    Args:
        url: URL to download from

    Returns:
        File content as bytes

    Raises:
        HTTPException: If file is too large or wrong content type
        requests.RequestException: If download fails
    """
    from fastapi import HTTPException

    MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

    # HEAD request first — check size and content-type before downloading
    try:
        head = requests.head(url, timeout=10, allow_redirects=True)
        head.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"HEAD request failed for URL, proceeding with GET: {e}")
        head = None

    if head is not None:
        content_length = head.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {int(content_length) // (1024*1024)}MB exceeds 20MB limit."
            )

        content_type = head.headers.get("Content-Type", "")
        if content_type and "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: expected PDF, got '{content_type}'."
            )

    logger.info(f"Downloading file from: {url[:50]}...")
    response = requests.get(url, timeout=settings.PDF_DOWNLOAD_TIMEOUT, stream=True)
    response.raise_for_status()

    # Double-check actual download size (Content-Length might be missing on HEAD)
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=65536):
        total += len(chunk)
        if total > MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: exceeds 20MB limit."
            )
        chunks.append(chunk)

    content = b"".join(chunks)
    logger.info(f"Downloaded {len(content)} bytes")
    return content


def pdf_to_images(pdf_bytes: bytes, max_pages: int | None = None) -> list[bytes]:
    """
    Convert PDF pages to JPEG images.
    
    Args:
        pdf_bytes: PDF file content
        max_pages: Optional limit on pages to convert
        
    Returns:
        List of JPEG image bytes
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    total_pages = len(doc)
    pages_to_process = total_pages if max_pages is None else min(total_pages, max_pages)
    
    logger.info(f"Converting {pages_to_process}/{total_pages} PDF pages to images")

    for page_num in range(pages_to_process):
        page = doc[page_num]
        matrix = fitz.Matrix(settings.PDF_RENDER_SCALE, settings.PDF_RENDER_SCALE)
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        images.append(buf.getvalue())

    doc.close()
    logger.info(f"Converted {len(images)} pages to images")
    return images


def images_to_base64(images: list[bytes]) -> list[dict]:
    """
    Convert image bytes to OpenAI-compatible content format.
    
    Args:
        images: List of image bytes
        
    Returns:
        OpenAI content array with base64 encoded images
    """
    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}"
            }
        })
    return content


def image_to_base64(image_bytes: bytes) -> str:
    """
    Convert single image bytes to base64 string.
    
    Args:
        image_bytes: Image content
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image_bytes).decode()
