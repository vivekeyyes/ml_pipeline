# model/test.py
THRESHOLD = 0.7  # Define an accuracy threshold

# Read the saved accuracy
with open("model/accuracy.txt", "r") as f:
    accuracy = float(f.read())

# Check if the model's accuracy meets the threshold
if accuracy < THRESHOLD:
    raise ValueError(f"Model accuracy {accuracy} is below the threshold of {THRESHOLD}")
else:
    print(f"Model passed with accuracy {accuracy}")
