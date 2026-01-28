import os
import sys

def validate_env():
    required_vars = [
        "DATABASE_URL",
        "BETTER_AUTH_SECRET",
        "BETTER_AUTH_URL"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("Please add them to your .env file.")
        sys.exit(1)
    
    print("Environment variables validated successfully!")

if __name__ == "__main__":
    validate_env()
