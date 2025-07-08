import json
import os

BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, 'objects.json')) as f:
    OBJECTS_MAP = json.load(f)
