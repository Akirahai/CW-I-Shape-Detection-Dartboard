**Hough Integration Guide**

This guide explains how to reuse your Lab 3 Hough implementation to improve dartboard detection in the `CW-I-Shape-Detection-Dartboard` assignment. It shows how to implement a basic custom Hough Transform for circles (and lines optionally), combine Hough evidence with Viola-Jones detections in `dartboard.py`, and evaluate the improved detector.

**Goal**: use at least one custom Hough Transform (implemented by you) to detect circular/elliptical structures of dartboards and combine those detections with the existing cascade output to improve precision and recall.

**Overview**:
- **Edge evidence**: compute gradients (Sobel) and an edge map + orientation (reuse `sobelEdge` from `Lab3-Coin-Counter-Challenge/Hough.py`).
- **Hough accumulator**: implement a circle Hough accumulator (3D: x, y, r) or optimized 2D accumulation using gradient orientation to vote for centers.
- **Peak extraction**: find peaks in Hough accumulator to produce circle hypotheses (x, y, r).
- **Fusion**: combine cascade detections (`faces` returned in `dartboard.py`) with circle hypotheses: (a) validate cascade boxes by presence of circle center inside or sufficient overlap, (b) add new detections where a strong circle exists but the cascade missed.
- **Evaluation**: write per-image detection files (as the current pipeline does) and regenerate `results/table.txt`.

**Where to put things**:
- Add Hough helper functions into a new module file `CW-I-Shape-Detection-Dartboard/hough_helpers.py` (or copy required functions into `dartboard.py`).
- Call Hough functions from `process_evaluation()` in `dartboard.py` after loading the image and before evaluation.

**Key functions (skeletons)**

1) Edge + orientation (reuse from Lab3)

```python
# returns (edgemap_bool, anglemap_float, magnitude_float)
def compute_edges_and_orientation(gray_image):
    # Reuse the sobelEdge implementation from Lab3 Hough.py
    edgex, edgey = sobelEdge(gray_image)
    magnitude = np.sqrt(edgex**2 + edgey**2)
    norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-12)
    anglemap = np.arctan2(edgey, edgex)
    edgemap = norm > 0.2  # tune threshold
    return edgemap, anglemap, magnitude
```

2) Circle Hough accumulator using gradient orientation (recommended)

```python
def hough_circle_accumulator(edgemap, anglemap, rmin=20, rmax=100):
    H = np.zeros((edgemap.shape[0], edgemap.shape[1], rmax - rmin + 1), dtype=np.uint32)
    ys, xs = np.nonzero(edgemap)
    for (y, x) in zip(ys, xs):
        theta = anglemap[y, x]
        # vote for centers along the normal to the edge (both directions)
        for r in range(rmin, rmax + 1):
            cx = int(round(x - r * np.cos(theta)))
            cy = int(round(y - r * np.sin(theta)))
            if 0 <= cx < H.shape[1] and 0 <= cy < H.shape[0]:
                H[cy, cx, r - rmin] += 1
            cx2 = int(round(x + r * np.cos(theta)))
            cy2 = int(round(y + r * np.sin(theta)))
            if 0 <= cx2 < H.shape[1] and 0 <= cy2 < H.shape[0]:
                H[cy2, cx2, r - rmin] += 1
    return H
```

3) Peak detection in accumulator

```python
def find_hough_peaks(H, rmin=20, threshold=30, neighborhood=8):
    # H: (H, W, R)
    circles = []
    H_copy = H.copy()
    while True:
        idx = np.unravel_index(np.argmax(H_copy), H_copy.shape)
        maxval = H_copy[idx]
        if maxval < threshold:
            break
        cy, cx, r_idx = idx
        circles.append((cx, cy, r_idx + rmin, int(maxval)))
        # suppress neighborhood
        r0 = max(0, r_idx - 1)
        r1 = min(H_copy.shape[2], r_idx + 2)
        y0 = max(0, cy - neighborhood)
        y1 = min(H_copy.shape[0], cy + neighborhood + 1)
        x0 = max(0, cx - neighborhood)
        x1 = min(H_copy.shape[1], cx + neighborhood + 1)
        H_copy[y0:y1, x0:x1, r0:r1] = 0
    return circles
```

4) Fusion: refine cascade boxes using circles

High-level rules to combine evidence (examples):
- If a cascade box contains a circle center and the circle radius approximately matches the box size (e.g., r ~ box_width/2), mark detection as validated.
- If a cascade box has no circle support, lower its confidence or mark for deletion (reduce FP).
- If a strong circle exists with no cascade box nearby, add a detection box centered on the circle (covers missed detections).

Example helper:

```python
def circle_to_box(circle):
    cx, cy, r, votes = circle
    x = cx - r
    y = cy - r
    w = 2 * r
    h = 2 * r
    return (x, y, w, h)

def refine_cascade_with_circles(cascade_boxes, circles, iou_thresh=0.3, match_radius_factor=0.6):
    kept = []
    # convert circles to boxes
    circle_boxes = [circle_to_box(c) for c in circles]
    for cb in cascade_boxes:
        x, y, w, h = cb
        cx = x + w/2
        cy = y + h/2
        validated = False
        for (ccx, ccy, r, votes) in circles:
            if (abs(ccx - cx) < w * match_radius_factor) and (abs(ccy - cy) < h * match_radius_factor):
                # radius must be roughly similar
                if r > 0 and (abs(2*r - max(w, h)) / max(w, h) < 0.6):
                    validated = True
                    break
        if validated:
            kept.append(cb)
    # add strong circles that are not matched to cascade boxes
    for cbox in circle_boxes:
        overlaps = [computeIOU(cbox, cb) for cb in cascade_boxes]
        if not any([o > iou_thresh for o in overlaps]):
            kept.append(cbox)
    return kept
```

**Integration steps in `dartboard.py`**
1. Import or paste the helper functions at top-level: `compute_edges_and_orientation`, `hough_circle_accumulator`, `find_hough_peaks`, `refine_cascade_with_circles`.
2. Inside `process_evaluation()` after reading `frame`:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
edgemap, anglemap, mag = compute_edges_and_orientation(gray)
H = hough_circle_accumulator(edgemap, anglemap, rmin=20, rmax=100)
circles = find_hough_peaks(H, rmin=20, threshold=25, neighborhood=10)
# refine the cascade detections
faces_refined = refine_cascade_with_circles(faces, circles)
# use faces_refined for evaluation and saving
```

3. Replace writes that currently use `faces` for evaluation with `faces_refined`.

4. Save an optional visualization image overlaying both cascade boxes and circle detections. Draw circles using `cv2.circle()` and boxes using `cv2.rectangle()`.

**Evaluation notes**
- Keep the same per-image `.txt` format (TP, FP, FN, Precision, Recall, F1_score). The `evaluation()` helper in `dartboard.py` expects bounding boxes as `(x,y,w,h)`; ensure circles are converted to boxes when needed.
- After processing all images, call `draw_table(results_dir='results')` (already present) to generate `results/table.txt` with the new results.

**Tuning tips**
- The important thresholds: edge threshold when building `edgemap` (e.g., 0.15-0.3), Hough vote threshold for peaks (e.g., 20-50), radius range `rmin/rmax` to match expected dartboard sizes in your dataset.
- Use a smaller `neighborhood` suppression for crowded scenes; larger for noisy accumulators.
- Consider smoothing the image (Gaussian blur) before computing gradients to reduce spurious edges.

**Performance considerations**
- The naive 3D accumulator is O(#edges * (rmax-rmin)) and can be slow for large images; use downsampling of the image / smaller radius range to speed up.
- You can also implement a 2D center-only accumulator by using gradient orientation to vote a single center per edge pixel per radius or use a Hough gradient approach (as above) to reduce cost.

**Optional: Hough for lines / ellipse evidence**
- Dartboards have circular rings and sometimes radial spokes; detect strong straight lines with your own Hough-line accumulator (or adapt OpenCV `HoughLines`) to add supporting evidence. For example, a high concentration of lines crossing near the same center supports a dartboard hypothesis.

**Quick checklist before running**
- [ ] Add `hough_helpers.py` with the code above (or embed in `dartboard.py`).
- [ ] Import helpers and call them in `process_evaluation()` replacing `faces` with refined detections when evaluating.
- [ ] Save visualization images to `results/` to inspect failures and successes.
- [ ] Tune thresholds on a small subset of images before running on the full dataset.

If you want, I can:
- create the `hough_helpers.py` file with these functions and apply a first integration patch to `dartboard.py` (I will not change evaluation code semantics, only replace which boxes are evaluated). Run it on one image and produce sample outputs.

Good luck — paste any errors or sample output and I will help tune parameters and fix integration issues.
