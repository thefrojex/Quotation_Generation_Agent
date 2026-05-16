import base64
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI


def analyze_media_with_openai(
    file_path: str,
    prompt: str,
    model: Optional[str] = None,
    detail: str = "auto"
) -> str:
    """
    Analyze an image or PDF file using OpenAI's vision API.
    
    Args:
        file_path: Path to the image or PDF file
        prompt: The prompt/question to ask about the media
        model: OpenAI model to use (defaults to env var OPENAI_MODEL or "gpt-4o")
        detail: Level of detail for vision analysis - "low", "high", or "auto"
    
    Returns:
        The AI's response text
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
        Exception: If the OpenAI API call fails
    """
    
    # Initialize OpenAI client (uses OPENAI_API_KEY env var)
    client = OpenAI()
    
    # Set default model
    if model is None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Validate file exists
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = file_path_obj.suffix.lower()
    
    # Supported image formats
    image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    if file_extension in image_formats:
        # Handle images directly
        return _analyze_image(client, file_path, prompt, model, detail)
    
    elif file_extension == '.pdf':
        # Handle PDFs
        return _analyze_pdf(client, file_path, prompt, model, detail)
    
    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            f"Supported formats: {image_formats}, .pdf"
        )


def _analyze_image(
    client: OpenAI,
    file_path: str,
    prompt: str,
    model: str,
    detail: str
) -> str:
    """Analyze an image file using OpenAI vision API."""
    
    with open(file_path, "rb") as image_file:
        image_data = base64.standard_b64encode(image_file.read()).decode("utf-8")
    
    # Determine media type based on file extension
    file_ext = Path(file_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(file_ext, 'image/jpeg')
    
    # Call OpenAI API with vision capability
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )
    
    return response.content[0].text


def _analyze_pdf(
    client: OpenAI,
    file_path: str,
    prompt: str,
    model: str,
    detail: str
) -> str:
    """Analyze a PDF file using OpenAI vision API."""
    
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF analysis. Install with: pip install pypdf")
    
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF analysis. "
            "Install with: pip install pdf2image"
        )
    
    # Convert PDF to images (first page by default)
    images = convert_from_path(file_path, first_page=1, last_page=1)
    
    if not images:
        raise ValueError(f"Could not extract images from PDF: {file_path}")
    
    # Convert PIL Image to base64
    import io
    image_byte_arr = io.BytesIO()
    images[0].save(image_byte_arr, format='PNG')
    image_data = base64.standard_b64encode(image_byte_arr.getvalue()).decode("utf-8")
    
    # Call OpenAI API with the PDF page as image
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )
    
    return response.content[0].text
