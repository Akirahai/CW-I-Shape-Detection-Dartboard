################################################
#
# COMS30068 - LeVietHai - Task2_dartboard.py
# University of Bristol
#
################################################

import os
import cv2
import sys
import argparse
import numpy as np
import statistics
from glob import glob
from utils import GroundTruthLoader, Evaluator, ResultWriter
from Task1_dartboard import Detector


class HoughCircleDetector:
    def __init__(self, r_min, r_max, threshold):
        self.r_min = r_min
        self.r_max = r_max
        self.threshold = threshold

    def sobelEdge(self,input):
        # intialise the output using the input
        edgeOutputX = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
        edgeOutputY = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
        # create the Gaussian kernel in 1D
        kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        kernelY = kernelX.T
        # we need to create a padded version of the input
        # or there will be border effects
        kernelRadiusX = round((kernelX.shape[0] - 1) / 2)
        kernelRadiusY = round((kernelX.shape[1] - 1) / 2)
        paddedInput = cv2.copyMakeBorder(input,
            kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
            cv2.BORDER_REPLICATE)
        # now we can do the convoltion
        for i in range(0, input.shape[0]):
            for j in range(0, input.shape[1]):
                patch = paddedInput[i:i+kernelX.shape[0], j:j+kernelX.shape[1]]
                edgeOutputX[i, j] = (np.multiply(patch, kernelX)).sum()
                edgeOutputY[i, j] = (np.multiply(patch, kernelY)).sum()
        return edgeOutputX, edgeOutputY
    
    
    def draw_circel(self, image, circles):
        imagewithcircle = image.copy()
        for circle_parameters in circles:
            imagewithcircle = cv2.circle(imagewithcircle,
                                        (circle_parameters[0], circle_parameters[1]),
                                        int(circle_parameters[2]),
                                        color=(0, 0, 255),
                                        thickness=2)
        return imagewithcircle
    

    def hough_circel(self, image, output_dir):
        print("Output directory for Hough Circle Detection:", output_dir)
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image.astype(np.float32)

        # apply sobel
        edgemapX, edgemapY = self.sobelEdge(gray_image)
        # magnitude
        magnitude = np.sqrt(edgemapX**2 + edgemapY**2)
        normmagnitude = (magnitude-magnitude.min())/(magnitude.max()-magnitude.min())
        # orientation
        anglemap = np.arctan2(edgemapY, edgemapX)
        # edge
        edgemap = normmagnitude > 0.2

        hough3D = np.zeros([image.shape[0], image.shape[1], self.r_max-self.r_min+1], dtype=np.float32)
        for i in range(0, image.shape[0]):  # go through all rows (or scanlines)
            for j in range(0, image.shape[1]):
                if edgemap[i, j] > 0:
                    for r in range(self.r_min, self.r_max+1):
                        x = (j + np.array([-1, 1])*r*np.cos(anglemap[i, j])).astype(int)
                        y = (i + np.array([-1, 1])*r*np.sin(anglemap[i, j])).astype(int)
                        for k in range(0, 2):
                            if (y[k] >= 0) and (y[k] < image.shape[0]) and (x[k] >= 0) and (x[k] < image.shape[1]):
                                hough3D[y[k], x[k], r-self.r_min] += 1

        hough2D = np.sum(hough3D, axis=2)

        circle_parameters_ls = []
        for i in range(0, hough3D.shape[0]):
            for j in range(0, hough3D.shape[1]):
                for k in range(0, hough3D.shape[2]):
                    if hough3D[i, j, k] >= self.threshold:
                        circle_parameters_ls.append([j, i, k + self.r_min])


        if len(circle_parameters_ls) > 0:
            output_dir = f"{output_dir}_HaveCircle"
            os.makedirs(output_dir, exist_ok=True)
            print(f"Detected {len(circle_parameters_ls)} circles using Hough Circle Transform.")
            image_with_circles = self.draw_circel(image, circle_parameters_ls)
            cv2.imwrite(os.path.join(output_dir, "imagewithcircle.jpg"), image_with_circles)
        else:
            output_dir = f"{output_dir}_NoCircle"
            os.makedirs(output_dir, exist_ok=True)
            print("No circles detected using Hough Circle Transform.")
        
        cv2.imwrite(os.path.join(output_dir, "edgemapX.jpg"), (edgemapX-edgemapX.min())/(edgemapX.max()-edgemapX.min())*255)
        cv2.imwrite(os.path.join(output_dir, "edgemapY.jpg"), (edgemapY-edgemapY.min())/(edgemapY.max()-edgemapY.min())*255)
        cv2.imwrite(os.path.join(output_dir, "hough2D.jpg"), (hough2D-hough2D.min())/(hough2D.max()-hough2D.min())*255)
        cv2.imwrite(os.path.join(output_dir, "magnitude.jpg"), normmagnitude*255)

        return circle_parameters_ls
    




class Detector_Hough(Detector):
    def __init__(self, r_min, r_max, threshold, cascade_path="Dartboardcascade/cascade.xml", border_size=10):
        super().__init__(cascade_path)
        self.border_size = border_size
        self.r_min = r_min
        self.r_max = r_max
        self.threshold = threshold
        self.hough_detector = HoughCircleDetector(r_min, r_max, threshold)


    def detect(self, image, image_root=""):
        #1. Use Cascade Classifier Detection
        objects = super().detect(image)
        print(f"Detected {len(objects)} objects using Cascade Classifier.")
       
        #2. Crop detected regions with border size
        crop_regions = {}
        for (x, y, w, h) in objects:
            x1 = np.clip(x - self.border_size, 0, image.shape[1])
            y1 = np.clip(y - self.border_size, 0, image.shape[0])
            x2 = np.clip(x + w + self.border_size, 0, image.shape[1])
            y2 = np.clip(y + h + self.border_size, 0, image.shape[0])
            crop = image[y1:y2, x1:x2]
            crop_regions[(x,y,w,h)] = crop

        #3. Apply Hough Circle Detection on cropped regions
        final_objects = []
        # Assume that if any circle is detected in the region, we keep the object as it is considered as a dartboard
        i = 0
        for object_box, region in crop_regions.items():
            i += 1
            circle_params_list = self.hough_detector.hough_circel(region, output_dir=f"{image_root}_object{i}")
            if len(circle_params_list) >= 2:  # At least 2 circles detected
                final_objects.append(object_box)
        print(f"After Hough Circle Detection, {len(final_objects)} objects remain as dartboards.")
            
        return final_objects

# ==========================================================
#                      Main Pipeline
# ==========================================================
class DartboardPipeline:
    def __init__(self, data_path, results_dir):
        self.data_path = data_path
        self.gt_loader = GroundTruthLoader()
        self.detector = Detector_Hough(r_min=10, r_max=100, threshold=15)
        self.evaluator = Evaluator()
        self.writer = ResultWriter(results_dir)

    def process_image_evaluation(self, img_path):
        image_root = os.path.splitext(os.path.basename(img_path))[0]

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Cannot read {img_path}")
            return

        detected = self.detector.detect(frame, image_root=f"{self.writer.results_dir}/{image_root}")
        gt_boxes = self.gt_loader.get(image_root)

        # Draw detection (green)
        for (x, y, w, h) in detected:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # Draw GT (red)
        for (x, y, w, h) in gt_boxes:
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)

        eval_dict = self.evaluator.evaluate(gt_boxes, detected)

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
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    pipeline = DartboardPipeline(data_path=args.data_path, results_dir=args.results_dir)
    pipeline.run()
