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
OUT_IMG = "dataset/images"
OUT_LABEL = "dataset/labels"

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
#          RANDOM IMAGE AUGMENTATIONS
# --------------------------------------------

def random_brightness_contrast(image):
    alpha = 1.0 + random.uniform(-0.4, 0.4)   # contrast
    beta = random.uniform(-50, 50)            # brightness
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def random_color_jitter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= random.uniform(0.7, 1.3)  # saturation
    hsv[:,:,2] *= random.uniform(0.7, 1.3)  # value
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
# Place Augmented Object into Background Image
# --------------------------------------------

def place_object(bg, obj):
    oh, ow = obj.shape[:2]
    bh, bw = bg.shape[:2]

    # random scale relative to background
    scale = random.uniform(0.15, 0.5)
    new_w = int(ow * scale)
    new_h = int(oh * scale)
    obj = cv2.resize(obj, (new_w, new_h))
    oh, ow = obj.shape[:2]

    # random valid position
    x = random.randint(0, bw - ow - 1)
    y = random.randint(0, bh - oh - 1)

    bg[y:y+oh, x:x+ow] = obj

    # YOLO normalized label
    cx = (x + ow / 2) / bw
    cy = (y + oh / 2) / bh
    w = ow / bw
    h = oh / bh

    return bg, (cx, cy, w, h)


# --------------------------------------------
#              MAIN GENERATION LOOP
# --------------------------------------------

count = 2000  # number of images to generate

for i in range(count):

    # --- Load random background ---
    bg = cv2.imread(random.choice(negatives))
    bg = cv2.resize(bg, (640, 480))

    # --- Augment the dartboard ---
    obj = pos.copy()
    obj = random_affine(obj)
    obj = random_perspective(obj)
    obj = random_brightness_contrast(obj)
    obj = random_color_jitter(obj)
    obj = random_blur(obj)
    obj = random_noise(obj)

    # --- Place dartboard into background ---
    composite, box = place_object(bg, obj)

    # --- Save image ---
    img_path = f"{OUT_IMG}/dart_{i}.jpg"
    lbl_path = f"{OUT_LABEL}/dart_{i}.txt"
    cv2.imwrite(img_path, composite)

    # --- Save YOLO label ---
    with open(lbl_path, "w") as f:
        f.write(f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n")

    if i % 50 == 0:
        print(f"Generated {i}/{count}...")


print("Dataset generation complete!")
