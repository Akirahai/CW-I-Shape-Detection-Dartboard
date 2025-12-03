################################################
# COMS30068 - LeVietHai - utils.py
# University of Bristol
################################################

import os
import cv2
import sys
import argparse
import numpy as np
import statistics
from glob import glob


# ==========================================================
#                  Ground Truth Loader
# ==========================================================
class GroundTruthLoader:
    def __init__(self, filepath="groundtruth/groundtruth.txt"):
        self.filepath = filepath
        self.data = self._load()

    def _load(self):
        groundtruth = {}
        with open(self.filepath) as f:
            for line in f.readlines():
                img, x, y, w, h = line.strip().split(',')
                if img not in groundtruth:
                    groundtruth[img] = []
                groundtruth[img].append((float(x), float(y), float(w), float(h)))
        return groundtruth

    def get(self, image_root):
        # Handle special cases used in your current code
        if image_root == "face6":
            return self.data.get("face1", [])
        return self.data.get(image_root, [])


# ==========================================================
#                        Evaluator
# ==========================================================
class Evaluator:
    @staticmethod
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xA = max(x1, x2)
        yA = max(y1, y2)
        xB = min(x1 + w1, x2 + w2)
        yB = min(y1 + h1, y2 + h2)

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH

        area1 = w1 * h1
        area2 = w2 * h2

        union = area1 + area2 - interArea
        return interArea / union if union > 0 else 0

    def evaluate(self, gt_boxes, detected_boxes, iou_threshold=0.3):
        TP = FP = FN = 0

        for gt in gt_boxes:
            matched = False
            for det in detected_boxes:
                if self.compute_iou(gt, det) >= iou_threshold:
                    TP += 1
                    matched = True
                    break
            if not matched:
                FN += 1

        FP = len(detected_boxes) - TP

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1
        }


# ==========================================================
#                     Result Writer
# ==========================================================
class ResultWriter:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_image(self, image_root, image):
        out_path = os.path.join(self.results_dir, f"detected_{image_root}.jpg")
        cv2.imwrite(out_path, image)

    def save_text(self, image_root, eval_dict, num_detected, num_gt):
        out_path = os.path.join(self.results_dir, f"detected_{image_root}.txt")
        with open(out_path, "w") as f:
            f.write(f"Image {image_root} with {num_detected} detected\n")
            f.write(f"Ground Truth Count: {num_gt}\n")
            for k, v in eval_dict.items():
                f.write(f"{k}: {v}\n")

    def generate_summary(self):
        files = sorted(glob(os.path.join(self.results_dir, 'detected_*.txt')))
        if not files:
            print("No detection files found.")
            return

        rows = []
        total_TP = total_FP = total_FN = 0

        for p in files:
            with open(p, 'r') as f:
                lines = [l.strip() for l in f.readlines()]

            def val(prefix):
                for L in lines:
                    if L.startswith(prefix):
                        try:
                            return float(L.split(':')[-1].strip())
                        except:
                            return 0.0
                return 0.0

            image = os.path.basename(p).replace('detected_', '').replace('.txt', '')
            TP = int(val("TP"))
            FP = int(val("FP"))
            FN = int(val("FN"))
            precision = val("Precision")
            recall = val("Recall")
            f1 = val("F1_score")

            rows.append((image, TP, FP, FN, precision, recall, f1))
            total_TP += TP
            total_FP += FP
            total_FN += FN

        mean_recall = statistics.mean([r[5] for r in rows])
        mean_f1 = statistics.mean([r[6] for r in rows])

        micro_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
        micro_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
        micro_f1 = (2 * total_TP) / (2 * total_TP + total_FP + total_FN)

        # Write summary
        out_path = os.path.join(self.results_dir, 'table.txt')
        with open(out_path, "w") as out:
            out.write("| Image | TP | FP | FN | Precision | Recall | F1 |\n")
            out.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

            for r in rows:
                out.write(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:.4f} |\n")

            out.write("\n")
            out.write(f"Mean Recall (TPR): {mean_recall:.4f}\n")
            out.write(f"Mean F1: {mean_f1:.4f}\n\n")
            out.write(f"Micro Precision: {micro_precision:.4f}\n")
            out.write(f"Micro Recall: {micro_recall:.4f}\n")
            out.write(f"Micro F1: {micro_f1:.4f}\n")

        print("Summary written to:", out_path)