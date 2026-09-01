import json
import sys

MIN_ACCURACY = 0.80

with open("metrics/metrics.json") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print(f"Accuracy: {accuracy:.3f}")
print(f"Required: {MIN_ACCURACY:.3f}")

if accuracy < MIN_ACCURACY:
    print("Model quality gate failed.")
    sys.exit(1)

print("Model quality gate passed.")