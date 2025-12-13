import os

# Enable seed printing only when explicitly set to 1 in the environment
PRINT_SEEDS = os.getenv("PRINT_SEEDS") == "1"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "20"))
