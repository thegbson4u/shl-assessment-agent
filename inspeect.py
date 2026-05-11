import json
import re
from pprint import pprint

with open("data/assessments.json", "r", encoding="utf-8") as f:
    content = f.read()

content = re.sub(r'[\x00-\x1F\x7F]', '', content)

data = json.loads(content)

pprint(data[0])