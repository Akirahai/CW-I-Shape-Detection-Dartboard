import cv2
import os
import random
import numpy as np
from glob import glob

# ----------------------------
# Paths
# ----------------------------
POS_IMG = "dart.bmp"
NEG_DIR = "negatives"
OUT_IMG = "darknet/data/obj"
OUT_LABEL = "darknet/data/obj"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABEL, exist_ok=True)

# ----------------------------
# Load Positives and Negatives
# ----------------------------
pos = cv2.imread(POS_IMG, cv2.IMREAD_COLOR)
if pos is None:
    raise FileNotFoundError(f"ERROR: Cannot load {POS_IMG}")

negatives = glob(os.path.join(NEG_DIR, "*"))
if len(negatives) == 0:
    raise FileNotFoundError("ERROR: negatives folder is empty!")

# --------------------------------------------
# RANDOM AUGMENTATION FUNCTIONS
# --------------------------------------------

def random_brightness_contrast(image):
    alpha = 1.0 + random.uniform(-0.4, 0.4)
    beta = random.uniform(-50, 50)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_color_jitter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= random.uniform(0.7, 1.3)
    hsv[:,:,2] *= random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_blur(image):
    if random.random() < 0.4:
        k = random.choice([3, 5])
        return cv2.GaussianBlur(image, (k, k), 0)
    return image

def random_noise(image):
    if random.random() < 0.3:
        noise = np.random.randint(0, 30, image.shape, dtype=np.uint8)
        return cv2.add(image, noise)
    return image

def random_affine(image):
    h, w = image.shape[:2]
    angle = random.uniform(-25, 25)
    scale = random.uniform(0.4, 1.2)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, scale)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def random_perspective(image):
    h, w = image.shape[:2]
    shift = 0.1 * min(h, w)
    pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
    pts2 = np.float32([
        [random.uniform(0, shift), random.uniform(0, shift)],
        [w - random.uniform(0, shift), random.uniform(0, shift)],
        [random.uniform(0, shift), h - random.uniform(0, shift)],
        [w - random.uniform(0, shift), h - random.uniform(0, shift)],
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

# --------------------------------------------
# OBJECT PLACEMENT FUNCTION
# --------------------------------------------

def place_object(bg, obj):
    oh, ow = obj.shape[:2]
    bh, bw = bg.shape[:2]

    # random scale
    scale = random.uniform(0.15, 0.5)
    new_w = int(ow * scale)
    new_h = int(oh * scale)
    obj = cv2.resize(obj, (new_w, new_h))
    oh, ow = obj.shape[:2]

    # random valid location
    x = random.randint(0, bw - ow - 1)
    y = random.randint(0, bh - oh - 1)

    # paste object
    bg[y:y+oh, x:x+ow] = obj

    # YOLO normalized box
    cx = (x + ow/2) / bw
    cy = (y + oh/2) / bh
    w = ow / bw
    h = oh / bh

    return bg, (cx, cy, w, h)

# --------------------------------------------
# MAIN GENERATION LOOP (NOW SUPPORTS 2 OBJECTS)
# --------------------------------------------

count = 1000  # number of images to generate

for i in range(count):

    bg = cv2.imread(random.choice(negatives))
    bg = cv2.resize(bg, (640, 480))

    labels = []

    # --- ALWAYS place 1 dartboard ---
    obj = pos.copy()
    obj = random_affine(obj)
    obj = random_perspective(obj)
    obj = random_brightness_contrast(obj)
    obj = random_color_jitter(obj)
    obj = random_blur(obj)
    obj = random_noise(obj)

    bg, box1 = place_object(bg, obj)
    labels.append(box1)

    # --- 20% chance to place a SECOND dartboard ---
    if random.random() < 0.20:
        obj2 = pos.copy()
        obj2 = random_affine(obj2)
        obj2 = random_perspective(obj2)
        obj2 = random_brightness_contrast(obj2)
        obj2 = random_color_jitter(obj2)
        obj2 = random_blur(obj2)
        obj2 = random_noise(obj2)

        bg, box2 = place_object(bg, obj2)
        labels.append(box2)

    # Save image
    img_path = f"{OUT_IMG}/dart_{i}.jpg"
    lbl_path = f"{OUT_LABEL}/dart_{i}.txt"
    cv2.imwrite(img_path, bg)

    # Save YOLO labels (multiple boxes if 2 objects)
    with open(lbl_path, "w") as f:
        for cx, cy, w, h in labels:
            f.write(f"0 {cx} {cy} {w} {h}\n")

    if i % 50 == 0:
        print(f"Generated {i}/{count}...")

print("Dataset generation complete!")
