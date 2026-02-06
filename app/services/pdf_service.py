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
    Download a file from URL.
    
    Args:
        url: URL to download from
        
    Returns:
        File content as bytes
        
    Raises:
        requests.RequestException: If download fails
    """
    logger.info(f"Downloading file from: {url[:50]}...")
    response = requests.get(url, timeout=settings.PDF_DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    logger.info(f"Downloaded {len(response.content)} bytes")
    return response.content


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
