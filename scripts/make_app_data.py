"""Regenerate app/data.js from the historical viewer artifact."""
import json
import os

os.makedirs("app", exist_ok=True)
d = json.load(open("out/proofs.json", encoding="utf-8"))
with open("app/data.js", "w", encoding="utf-8") as f:
    f.write("window.PROOF_DATA = ")
    json.dump(d, f)
    f.write(";\n")
print("wrote app/data.js", os.path.getsize("app/data.js"), "bytes")
