from rag_backend.config import settings
import os
from dotenv import load_dotenv

load_dotenv()

print("="*60)
print("DEBUG ENV VARIABLES")
print("="*60)
print(f"OS ENV NEON_DATABASE_URL: {os.getenv('NEON_DATABASE_URL')}")
print(f"SETTINGS NEON_DATABASE_URL: {settings.neon_database_url}")
print("="*60)
