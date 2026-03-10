import base64
from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_VISION_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def image_bytes_to_data_url(image_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def summarize_page_image(image_bytes: bytes) -> str:
    data_url = image_bytes_to_data_url(image_bytes)

    response = client.chat.completions.create(
        model=OPENAI_VISION_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are extracting visual information from an industrial document page. "
                    "Describe only useful document content such as diagrams, warning symbols, tables, labels, "
                    "flow arrows, equipment names, measurements, and any readable text. "
                    "Do not invent details. Be concise."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the important visual information on this document page."},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()