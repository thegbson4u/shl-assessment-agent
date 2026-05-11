import json

# Read raw file
with open("data/assessments.json", "r", encoding="utf-8") as f:
    content = f.read()

# Remove problematic control characters
content = content.replace("\n", " ")
content = content.replace("\r", " ")
content = content.replace("\t", " ")

# Load JSON safely
data = json.loads(content)

print("Total assessments:", len(data))

# Print first item
print(data[0])