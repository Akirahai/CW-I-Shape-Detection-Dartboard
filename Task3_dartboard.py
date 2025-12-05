################################################
# COMS30068 - LeVietHai - Task3_dartboard.py
# University of Bristol
################################################

import os
import cv2
import subprocess
import re
import argparse
from glob import glob
from utils import GroundTruthLoader, Evaluator, ResultWriter
# ==========================================================
#                       YOLO Detector
# ==========================================================
class YOLODetector:
    def __init__(self,
                 data_path="darknet/data/obj.data",
                 cfg_path="darknet/cfg/yolov4-tiny-dartboard.cfg",
                 weights_path="backup/yolov4-tiny-dartboard_best.weights",
                 darknet_binary="./darknet/darknet"):
        
        self.data_path = data_path
        self.cfg_path = cfg_path
        self.weights_path = weights_path
        self.darknet = darknet_binary

    def detect(self, image_path):
        """
        Runs Darknet and returns bounding boxes as (x, y, w, h)
        """
        cmd = [
            self.darknet,
            "detector", "test",
            self.data_path,
            self.cfg_path,
            self.weights_path,
            image_path,
            "-dont_show",
            "-ext_output",
            "> output.txt"
        ]


        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout

        detections = []

        # Pattern for lines like: dartboard: 98%
        class_conf_pattern = r"(\w+):\s*(\d+)%"
        # Pattern for bounding box coordinates
        box_pattern = r"left_x:\s*(-?\d+)\s*top_y:\s*(-?\d+)\s*width:\s*(\d+)\s*height:\s*(\d+)"

        class_conf_matches = re.findall(class_conf_pattern, output)
        box_matches = re.findall(box_pattern, output)

        boxes = []
        # Pair them together
        for (cls, conf), (x, y, w, h) in zip(class_conf_matches, box_matches):
            if float(conf) < 25.0:
                continue
            boxes.append((int(x), int(y), int(w), int(h)))

        return boxes

    


# ==========================================================
#                  Dartboard Evaluation Pipeline
# ==========================================================
class DartboardPipeline:
    def __init__(self, data_path, results_dir):
        self.data_path = data_path

        self.gt_loader = GroundTruthLoader()
        self.detector = YOLODetector()

        self.evaluator = Evaluator()
        self.writer = ResultWriter(results_dir)

    def process_image_evaluation(self, img_path):
        image_root = os.path.splitext(os.path.basename(img_path))[0]

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Cannot read {img_path}")
            return

        detected = self.detector.detect(img_path)  # YOLO detection
        gt_boxes = self.gt_loader.get(image_root)

        # Draw detections (GREEN)
        for (x, y, w, h) in detected:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Draw Ground Truth (RED)
        for (x, y, w, h) in gt_boxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)

        # Evaluate detections
        eval_dict = self.evaluator.evaluate(gt_boxes, detected)

        # Save result image + text
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
    parser.add_argument("--results_dir", type=str, default="Task3_results")
    args = parser.parse_args()

    pipeline = DartboardPipeline(data_path=args.data_path, results_dir=args.results_dir)
    pipeline.run()
