import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

INDEX_DIR = os.getenv("INDEX_DIR", "./indexes")

TOP_K = int(os.getenv("TOP_K", 5))
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", 12))
SPARSE_CANDIDATES = int(os.getenv("SPARSE_CANDIDATES", 12))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", 0.65))

MAX_CHARS_PER_CHUNK = int(os.getenv("MAX_CHARS_PER_CHUNK", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

ENABLE_VISION_SUMMARY = os.getenv("ENABLE_VISION_SUMMARY", "true").lower() == "true"
VISION_PAGE_TEXT_THRESHOLD = int(os.getenv("VISION_PAGE_TEXT_THRESHOLD", 120))

SOURCE_CONFIG = {
    "safety": {
        "label": "Safety Procedures",
        "file_path": "./safety_procedures.txt",
        "description": "Workplace safety procedures, PPE rules, emergency response, lockout-tagout, spills, hazards, and safe operating practices."
    },
    "maintenance": {
        "label": "Maintenance Manuals",
        "file_path": "./maintenance_manuals.txt",
        "description": "Machine troubleshooting, preventive maintenance, repair guidance, calibration, equipment checks, and maintenance instructions."
    },
    "quality": {
        "label": "Quality Control Standards",
        "file_path": "./quality_control_standards.txt",
        "description": "Inspection standards, acceptance criteria, defect classification, tolerances, process control, and quality procedures."
    },
}