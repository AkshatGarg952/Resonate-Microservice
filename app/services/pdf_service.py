"""
PDF processing service - download, convert to images.
"""
import io
import base64
import httpx
from PIL import Image
import fitz  # PyMuPDF

from app.core.config import settings
from app.core.logger import logger, log_error


# Maximum pages to render per PDF — prevents OOM under concurrent load
MAX_PAGES = 10


async def download_file(url: str) -> bytes:
    """
    Download a file from URL with safety guards (async).

    Uses httpx async client to avoid blocking the event loop.
    Rejects files > 20MB or non-PDF content types to prevent
    malicious URLs from crashing the process.

    Args:
        url: URL to download from

    Returns:
        File content as bytes

    Raises:
        HTTPException: If file is too large or wrong content type
        httpx.HTTPError: If download fails
    """
    from fastapi import HTTPException

    MAX_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

    async with httpx.AsyncClient(timeout=settings.PDF_DOWNLOAD_TIMEOUT) as client:
        # HEAD request first — check size and content-type before downloading
        try:
            head = await client.head(url, follow_redirects=True)
            head.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"HEAD request failed for URL, proceeding with GET: {e}")
            head = None

        if head is not None:
            content_length = head.headers.get("content-length")
            if content_length and int(content_length) > MAX_SIZE_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: {int(content_length) // (1024*1024)}MB exceeds 20MB limit."
                )

            content_type = head.headers.get("content-type", "")
            if content_type and "pdf" not in content_type.lower() and "octet-stream" not in content_type.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: expected PDF, got '{content_type}'."
                )

        logger.info(f"Downloading file from: {url[:50]}...")

        # Stream download — enforce size limit during transfer
        chunks = []
        total = 0
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=65536):
                total += len(chunk)
                if total > MAX_SIZE_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail="File too large: exceeds 20MB limit."
                    )
                chunks.append(chunk)

    content = b"".join(chunks)
    logger.info(f"Downloaded {len(content)} bytes")
    return content


def pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_PAGES) -> list[bytes]:
    """
    Convert PDF pages to JPEG images.

    Caps rendering at max_pages (default 10) to prevent OOM under load.
    Explicitly releases PyMuPDF pixmap memory after each page render.

    Args:
        pdf_bytes: PDF file content
        max_pages: Hard limit on pages to convert (default: MAX_PAGES)

    Returns:
        List of JPEG image bytes
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    total_pages = len(doc)
    pages_to_process = min(total_pages, max_pages)

    logger.info(f"Converting {pages_to_process}/{total_pages} PDF pages to images (cap={max_pages})")

    for page_num in range(pages_to_process):
        page = doc[page_num]
        matrix = fitz.Matrix(settings.PDF_RENDER_SCALE, settings.PDF_RENDER_SCALE)
        pix = page.get_pixmap(matrix=matrix)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Release pixmap memory immediately — don't hold all pages in RAM
        del pix

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        images.append(buf.getvalue())
        buf.close()

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
