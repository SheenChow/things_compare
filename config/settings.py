import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL_EXTRACT = "gemini-2.5-flash-lite"
GEMINI_MODEL_ANALYSIS = "gemini-3-flash-preview"
GEMINI_MODEL_IMAGE = "gemini-2.5-flash-image"

if not GEMINI_API_KEY:
    raise ValueError("请设置环境变量 GEMINI_API_KEY")
