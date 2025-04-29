from dotenv import load_dotenv
import os

def get_openai_api_key() -> str:
    load_dotenv()  
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return key

def get_serper_api_key() -> str:
    load_dotenv()  
    key = os.getenv("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY not set in .env")
    return key
