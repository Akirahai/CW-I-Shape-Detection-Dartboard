################################################
# COMS30068 - LeVietHai - Task1_dartboard.py
# University of Bristol
################################################

import os
import cv2
import sys
import argparse
import numpy as np
import statistics
from glob import glob
from utils import GroundTruthLoader, Evaluator, ResultWriter



# ==========================================================
#                       Detector
# ==========================================================
class Detector:
    def __init__(self, cascade_path="Dartboardcascade/cascade.xml"):
        self.model = cv2.CascadeClassifier(cascade_path)
        if self.model.empty():
            print("ERROR: Cascade model failed to load.")
            sys.exit(1)

    def detect(self, image):
        # 1. Prepare Image by turning it into Grayscale and normalising lighting
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # 2. Perform Viola-Jones Object Detection
        objects = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            flags=0,
            minSize=(10,10),
            maxSize=(300,300)
        )
        print(f"Detected {len(objects)} objects using Cascade Classifier.")

        return objects.tolist() if len(objects) else []


# ==========================================================
#                      Main Pipeline
# ==========================================================
class DartboardPipeline:
    def __init__(self, data_path, results_dir):
        self.data_path = data_path
        self.gt_loader = GroundTruthLoader()
        self.detector = Detector()
        self.evaluator = Evaluator()
        self.writer = ResultWriter(results_dir)

    def process_image_evaluation(self, img_path):
        image_root = os.path.splitext(os.path.basename(img_path))[0]

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Cannot read {img_path}")
            return

        detected = self.detector.detect(frame)
        gt_boxes = self.gt_loader.get(image_root)

        # Draw detection (green)
        for (x, y, w, h) in detected:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Draw GT (red)
        for (x, y, w, h) in gt_boxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)

        eval_dict = self.evaluator.evaluate(gt_boxes, detected)
        # print(f"Evaluation for {image_root}: {eval_dict}")

        # Save results
        self.writer.save_image(image_root, frame)
        self.writer.save_text(image_root, eval_dict, len(detected), len(gt_boxes))


    def run(self):
        for img_file in os.listdir(self.data_path):
            self.process_image_evaluation(os.path.join(self.data_path, img_file))

        self.writer.generate_summary()


# ==========================================================
#                          MAIN
# ==========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dartboard Detection Pipeline")
    parser.add_argument("--data_path", type=str, default="Dartboard")
    parser.add_argument("--results_dir", type=str, default="Task1_results")
    args = parser.parse_args()

    pipeline = DartboardPipeline(data_path=args.data_path, results_dir=args.results_dir)
    pipeline.run()
