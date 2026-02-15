import dotenv
import os

dotenv.load_dotenv()
print("Keys in .env:")
for key in os.environ:
    if "MEM0" in key or "API_KEY" in key:
        print(key)
