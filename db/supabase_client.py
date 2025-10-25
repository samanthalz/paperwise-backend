from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()  

SUPABASE_URL = os.getenv("PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials are missing!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
