# Subtask 1: The Dartboard Detector
## Dataset
- `dart.bmp`: dartboard image, serving as prototype for generating the whole set of training images
- `negatives`: folder contatining all the images in which there is no dartboard inside, `negatives.dat`: list all files name in the directory
- `dart.vec`: 500 tiny 20 x 20 images of dartboards, crated by randomly changing viewing angle and constrast.


## Step-by-step training process summary:
- Create training dataset:
```bash
opencv_createsamples -img dart.bmp -vec dart.vec  -w 20 -h 20 -num 500 -maxidev 80 -maxxangle 0.8 -maxyangle 0.8 -maxzangle 0.2
```
Each of these sample is created by randomly changing viewing angle and contrast (up to the maximum values specified) to reflect the possible variability of viewing parameters in real images better. -> `dart.vec`

- Process training databoard detector:
Use AdaBoost, Training Directory `Dartboardcascade`.
```bash
opencv_traincascade -data Dartboardcascade -vec dart.vec -bg negatives.dat -numPos 500 -numNeg 500 -numStages 3 -maxDepth 1 -w 20 -h 20 -minHitRate 0.999  -maxFalseAlarmRate 0.05 -mode ALL
```

- Parameter explanations
    - -data Dartboardcascade: Directory where the trained cascade files will be saved. It will create multiple XML files, and the final classifier is `cascade.xml`.
    - -vec dart.vec: Positive samples 
    - -bg negatives.dat: text files for paths to negative images
    - -numPos 500: Number of positive samples to use for training.
    - -numNeg 500: Number of negative samples to use for training.
    - -numStages 3: Number of cascade stages. 
    - -maxDepth 1: Maximum depth of each weak classifier (decisio tree). Depth 1 means simple decision stumps.
    - -w 20 -h 20: Size of detection window
    - -minHitRate 0.99: Min TPR = 0.99
    - -maxFalseAlarm 0.05: Max FPR = 0.05
    - -mode All: Use all features (Haar).

- Dataset Handle in Each Stage:

    - Positive Dataset (Fixed):In every stage, the algorithm uses the same 500 positive samples (dartboard images provided in dart.vec). These act as the ground truth for what the object looks like.
    - Negative Dataset (Dynamic - "Hard Negative Mining"):The negative set changes for every stage. The algorithm does not simply use 500 random background images. Instead, it extracts 500 specific $20\times20$ windows based on the failures of the previous stage.
        - Stage 0: Uses 500 random background windows.
        - Stage 1: (6% rate): To fill the bucket with 500 failures (FP), the computer had to check roughly 8,300 windows.
        - Stage 2:(0.8% rate): To fill the bucket with 500 failures (FP), the computer had to check roughly 57,000 windows.

        
- Results: 

After training, a file `cascade.xml` is obtained, can load in OpenCV for detections: `detector = cv2.CascadeClassifier('Dartboardcascade/cascade.xml')`

## Training Performance (Assignment 1)
- Training Logs. Training Performance Analysis are inside the folowing foler `training_performance_results.txt`. Plotting Code for this assignment is file `plot_training.py`.

![alt text](Task1_performance_results/training_stages.png)
**Figure 1:** Training Performance Graph on the training data for the three different stages.

### Interpretation

- The TPR remains constant at **1.00** across all stages. This indicates that the classifier successfully preserves all positive samples (dartboards) without sacrificing sensitivity as complexity increases to reduce FPR with the constraint `-minHitRate 0.99`

- The False Positive Rate (FPR) decreases significantly in every stage, correlating with the drastic drop in `acceptanceRatio` (0.06 in Stage 1 $\to$ 0.008 in Stage 2). This confirms that as the model progressed, it required scanning exponentially more background windows to locate the 500 "hard negatives" needed to fill the training bucket.

- Both Stage 1 and Stage 2 required exactly 6 weak classifiers to satisfy the max False Alarm Rate. However, their internal behavior differed:
    -  **Stage 1** showed significant volatility, with a sharp FPR spike between the 4th and 5th classifiers ($0.106 \to 0.502$), likely due to the introduction of a broad, aggressive feature.
    -  **Stage 2** displayed a more stable pattern ($0.386 \to 0.400$ at the same step), indicating a more consistent refinement process on the hardest dataset.
    -  Ultimately, both stages successfully converged to the required criteria ($FPR \le 0.05$).

## Testing Performance (Assignment 2)

- The code to test darboard's detector of all images inside `Dartboard` folder is `Task1_dartboard.py`. To test on that dataset, simply write `python Task1_dartboard.py --data_path Dartboard` in the terminal. Given results of this dataset is produced in `results` folder.
- **Groundtruth** used for this evaluation was annotated by human (me) using `annotate.py` code, result of groundtruth is produced inside `groundtruth.txt`. All was placed inside `groundtruth` folder.

- 3 examples of images

![alt text](Task1_results/detected_dart3.jpg)

**Figure 2:**: Examples with bouding boxes drawn for dart3.jpg

![alt text](Task1_results/detected_dart1.jpg)

**Figure 3:**: Examples with bouding boxes drawn for dart1.jpg

![alt text](Task1_results/detected_dart2.jpg)

**Figure 4:**: Examples with bouding boxes drawn for dart2.jpg

### Interpretation


| Image   | Recall (TPR) | F1     |
| :------ | :-----------: | :----: |
| dart0   | 1.0000        | 0.4000 |
| dart1   | 1.0000        | 0.6667 |
| dart2   | 1.0000        | 0.2500 |
| dart3   | 1.0000        | 0.4000 |
| dart4   | 0.0000        | 0.0000 |
| dart5   | 1.0000        | 0.1818 |
| dart6   | 0.0000        | 0.0000 |
| dart7   | 0.0000        | 0.0000 |
| dart8   | 1.0000        | 0.2667 |
| dart9   | 1.0000        | 0.5000 |
| dart10  | 0.0000        | 0.0000 |
| dart11  | 0.0000        | 0.0000 |
| dart12  | 0.0000        | 0.0000 |
| dart13  | 0.0000        | 0.0000 |
| dart14  | 1.0000        | 0.0488 |
| dart15  | 1.0000        | 0.6667 |

**Table 1:** Evaluation Result of each image across the 16 images

**Mean Recall (TPR):**: 0.5625

**Mean F1:** 0.2113

**Performance Discuss**
- The detector achieve full recall on 9 images, and mean recall across 16 images as 0.5625
- The detector also suffers from many false positives, reducing precision and F1

**Reasons for different TPR values compared to the performances achieved in a**
- Training used a single prototype `dart.bmp` to create synthetic positives (`dart.vec`): this limits positive variability and can cause the detector to lack some features caused by background area on the dartboard images.

- The cascade is shallow (-maxDepth 1), and the number of stages is small (-numStages 3), so the model may not have good model structure to realize correct pattern of dartboard images, and they may be overfitted to reject many background patterns when applied to whole test images.

