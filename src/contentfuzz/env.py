import os

# Enable seed printing only when explicitly set to 1 in the environment
PRINT_SEEDS = os.getenv("PRINT_SEEDS") == "1"
_max_retries = os.getenv("MAX_RETRIES")
MAX_RETRIES = int(_max_retries) if _max_retries else None
