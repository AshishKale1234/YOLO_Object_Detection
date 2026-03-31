# YOLO Object Detection using Darknet

Real-time multi-class object detection using YOLOv4 with the Darknet framework, CUDA-enabled GPU acceleration, and OpenCV for inference and visualization.

![Detection results](images/output/detection_grid.png)

---

## Overview

This project implements a GPU-accelerated object detection pipeline using YOLOv4 and the Darknet framework. The system detects and localizes 80 object categories from the COCO dataset in a single forward pass, with configurable confidence thresholding and Non-Max Suppression for precision-recall optimization.

---

## How to replicate

### 1. Clone the repository

    git clone https://github.com/AshishKale1234/YOLO_Object_Detection.git
    cd YOLO_Object_Detection

### 2. Set up environment

    conda create -n yolo python=3.10 -y
    conda activate yolo
    pip install -r requirements.txt

### 3. Install OpenCV C++ headers (required for Darknet compilation)

    sudo apt-get install -y libopencv-dev

### 4. Compile Darknet with CUDA and OpenCV

    git clone https://github.com/AlexeyAB/darknet
    cd darknet
    sed -i 's/GPU=0/GPU=1/' Makefile
    sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile
    make

Verify compilation:

    ls -lh libdarknet.so darknet

### 5. Download YOLOv4 weights and config

    mkdir -p weights cfg data

    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -O weights/yolov4.weights
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg -O cfg/yolov4.cfg
    wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names -O data/coco.names

### 6. Run detection

    python -c "
    import sys
    sys.path.append('./src')
    from detector import YOLODetector
    from visualizer import draw_detections, save_result
    import cv2

    detector = YOLODetector(
        cfg_path     = 'cfg/yolov4.cfg',
        weights_path = 'weights/yolov4.weights',
        names_path   = 'data/coco.names',
        conf_thresh  = 0.5,
        nms_thresh   = 0.4,
        use_gpu      = False,
    )

    image = cv2.imread('your_image.jpg')
    boxes, confidences, class_ids = detector.detect(image)
    result = draw_detections(image, boxes, confidences, class_ids,
                             detector.classes, detector.colors)
    save_result(result, 'output.jpg')
    print(f'{len(boxes)} objects detected')
    "

### 7. Google Colab walkthrough

Open `YOLO_Object_Detection.ipynb` in Colab for the full step-by-step walkthrough:

1. Go to colab.research.google.com
2. File -> Upload notebook -> select YOLO_Object_Detection.ipynb
3. Runtime -> Change runtime type -> T4 GPU
4. Run cells top to bottom

Note: Darknet must be recompiled each Colab session since /content is wiped on reset. The `apt-get install libopencv-dev` step is required before compilation.

---

## Architecture — YOLOv4

    Input image (any size)
        -> Resize to 416x416
        -> Normalize pixels [0,255] -> [0,1]
        -> CSPDarknet53 backbone (feature extraction)
        -> PANet neck (multi-scale feature fusion)
        -> 3 YOLO detection heads:
            yolo_139 -> small objects  (52x52 grid)
            yolo_150 -> medium objects (26x26 grid)
            yolo_161 -> large objects  (13x13 grid)
        -> Per cell: [cx, cy, w, h, objectness, 80 class scores]
        -> Confidence filtering (threshold = 0.5)
        -> Non-Max Suppression (IoU threshold = 0.4)
        -> Final detections: boxes + labels + scores

---

## Detection pipeline

Each detection goes through two filtering steps:

**Confidence thresholding** — any detection with `objectness x class_score < threshold` is discarded. Lower threshold = more detections but more false positives. Higher threshold = fewer but more reliable detections.

**Non-Max Suppression (NMS)** — when multiple boxes overlap the same object (IoU > nms_thresh), only the highest confidence box is kept. Lower NMS threshold = more aggressive merging. Higher = allows more overlapping boxes.

---

## Key implementation notes

- Darknet compiled with GPU=1 CUDNN=1 OPENCV=1 LIBSO=1 for full GPU acceleration
- OpenCV dnn module loads .cfg and .weights directly — no Python bindings needed
- OpenCV dnn CUDA backend unavailable on standard Colab builds — CPU inference used via OpenCV dnn, GPU inference confirmed via native Darknet CLI (127ms on T4)
- COCO class ant not in training set — demonstrates model operates within its training distribution
- YOLOv4 input normalized to 416x416 with BGR->RGB swap via blobFromImage

---

## Results

| Image | Detections | Top class | Confidence |
|-------|-----------|-----------|------------|
| dogs.jpg | 1 | dog | 0.98 |
| cat.jpg | 1 | cat | 0.90 |

---

## Project structure

    src/
        detector.py     # YOLODetector class — loads network, runs inference, applies NMS
        visualizer.py   # OpenCV drawing — boxes, labels, confidence scores
    images/
        output/         # Detection result visualizations
    YOLO_Object_Detection.ipynb  # Full Colab walkthrough
    requirements.txt
    README.md

Weights, cfg, data, and input images are not included. Download separately following setup instructions above.

---

## Dependencies

| Package | Version |
|---------|---------|
| opencv-python | >=4.7.0 |
| numpy | >=1.24.0 |
| torch | >=2.0.0 |
| matplotlib | >=3.7.0 |

---

## References

- YOLOv4: Optimal Speed and Accuracy of Object Detection — Bochkovskiy et al., 2020 — https://arxiv.org/abs/2004.10934
- Darknet framework — https://github.com/AlexeyAB/darknet
- COCO Dataset — https://cocodataset.org
