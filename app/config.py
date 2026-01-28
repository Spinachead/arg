import json
import dotenv
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(project_root, "app", ".env")
config_path = os.path.join(project_root, "app", "config.json")

dotenv.load_dotenv(dotenv_path)

with open(config_path, "r") as config_file:
    config = json.load(config_file)
