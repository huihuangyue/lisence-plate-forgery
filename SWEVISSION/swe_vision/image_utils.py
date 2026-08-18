"""
Image helper utilities for encoding, MIME type detection, and
building OpenAI-compatible image content parts.
"""

import base64
import mimetypes
import os
from io import BytesIO
from typing import Any, Dict


def image_file_to_base64(image_path: str, max_file_size: str = "20MB") -> str:
    """Convert image file to base64 data URI.

    If the file exceeds *max_file_size*, the image is progressively
    compressed as JPEG (lowering quality) until it fits.

    Parameters
    ----------
    image_path : str
        Path to an image file (PNG, JPG/JPEG, GIF, BMP, WEBP, TIFF ...).
    max_file_size : str or int or None
        Human-readable size string like ``"10MB"`` / ``"500KB"``, raw byte
        count as ``int``, or ``None`` to skip compression.

    Returns
    -------
    str
        A ``data:<mime>;base64,...`` URI.
    """
    from PIL import Image

    def _parse_size(size):
        if size is None:
            return None
        if isinstance(size, (int, float)):
            return int(size)
        size = size.strip().upper()
        units = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
        for suffix, factor in sorted(units.items(), key=lambda x: -len(x[0])):
            if size.endswith(suffix):
                return int(float(size[: -len(suffix)].strip()) * factor)
        return int(size)

    max_bytes = _parse_size(max_file_size)

    file_size = os.path.getsize(image_path)
    file_extension = os.path.splitext(image_path)[1].lower()

    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    mime_type = mime_map.get(file_extension)
    if mime_type is None:
        raise ValueError(
            f"Unsupported image format '{file_extension}'. "
            f"Supported: {', '.join(sorted(mime_map.keys()))}"
        )

    if max_bytes is None or file_size <= max_bytes:
        with open(image_path, "rb") as f:
            base64_string = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_string}"

    img = Image.open(image_path)
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    for quality in range(95, 5, -5):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if buf.tell() <= max_bytes:
            base64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_string}"

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=5, optimize=True)
    base64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_string}"


def guess_mime_type(file_path: str) -> str:
    """Guess MIME type from file extension."""
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "image/png"


def make_image_content_part(file_path: str) -> Dict[str, Any]:
    """Create an OpenAI image_url content part from a local file."""
    b64 = image_file_to_base64(file_path)
    return {
        "type": "image_url",
        "image_url": {
            "url": b64,
        },
    }


def make_base64_image_content_part(b64_data: str, mime: str = "image/png") -> Dict[str, Any]:
    """Create an OpenAI image_url content part from base64 data."""
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{b64_data}",
        },
    }
