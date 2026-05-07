import json
data = {"type": "token", "text": "\n"}
s = json.dumps(data)
print(repr(s))
