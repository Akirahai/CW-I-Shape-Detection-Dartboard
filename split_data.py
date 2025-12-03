import glob
import random
import os

files = glob.glob("dataset/images/*.jpg")
random.shuffle(files)

# Convert all paths to forward slashes
files = [p.replace("\\", "/") for p in files]

split = int(len(files) * 0.9)
train = files[:split]
valid = files[split:]

with open("yolo/train.txt", "w") as f:
    f.write("\n".join(train) + "\n")

with open("yolo/valid.txt", "w") as f:
    f.write("\n".join(valid) + "\n")
