# backend/config_loader.py
import yaml
import os

def load_config(filename="config.yaml"):
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', filename)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
