
import cv2
import numpy as np
from pathlib import Path


class YOLODetector:
    """
    YOLOv4 object detector using OpenCV's dnn module.

    OpenCV dnn loads Darknet's .cfg and .weights files directly
    and runs inference on GPU via CUDA backend — same weights,
    same architecture, full Python control over outputs.

    Args:
        cfg_path     : path to yolov4.cfg
        weights_path : path to yolov4.weights
        names_path   : path to coco.names
        conf_thresh  : confidence threshold — discard below this
        nms_thresh   : NMS IoU threshold — merge boxes above this
        input_size   : network input size (must match cfg width/height)
        use_gpu      : use CUDA backend if available
    """

    def __init__(self, cfg_path, weights_path, names_path,
                 conf_thresh=0.5, nms_thresh=0.4,
                 input_size=416, use_gpu=False):

        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.input_size  = input_size

        # ── Load class names ─────────────────────────────────────────────────
        with open(names_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(self.classes)} class names")

        # ── Load network ──────────────────────────────────────────────────────
        print("Loading YOLOv4 network via OpenCV dnn...")
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

        # ── Set backend ───────────────────────────────────────────────────────
        # CUDA backend uses GPU — same as Darknet's GPU=1 compilation
        # Falls back to CPU if CUDA not available
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Backend: CUDA (GPU)")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Backend: OpenCV (CPU)")

        # ── Get output layer names ────────────────────────────────────────────
        # YOLOv4 has 3 YOLO output layers — one for each detection scale
        # small objects (large feature map), medium, large objects (small map)
        layer_names        = self.net.getLayerNames()
        unconnected        = self.net.getUnconnectedOutLayers()
        self.output_layers = [layer_names[i - 1] for i in unconnected.flatten()]
        print(f"Output layers    : {self.output_layers}")

        # ── Fixed color per class for visualization ───────────────────────────
        np.random.seed(42)
        self.colors = np.random.randint(50, 230,
                                        size=(len(self.classes), 3),
                                        dtype=np.uint8)

        print("✓ YOLODetector ready\n")


    def preprocess(self, image):
        """
        Convert image to blob — the format YOLOv4 expects.

        blobFromImage does three things:
        1. Resizes image to input_size × input_size
        2. Normalizes pixel values from [0,255] to [0,1] (scale=1/255)
        3. Converts BGR → RGB (swapRB=True) since YOLO was trained on RGB

        Output shape: [1, 3, 416, 416]
        """
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor = 1/255.0,
            size        = (self.input_size, self.input_size),
            mean        = (0, 0, 0),
            swapRB      = True,
            crop        = False,
        )
        return blob


    def postprocess(self, outputs, img_h, img_w):
        """
        Parse raw YOLO outputs into boxes, confidences, class ids.

        Raw output from each YOLO layer:
            shape: [num_detections, 5 + num_classes]
            columns: [cx, cy, w, h, objectness, class_0, class_1, ...]

        cx, cy, w, h are normalized to [0,1] relative to image size.
        objectness = confidence that ANY object is present.
        class scores = probability of each specific class given object present.
        Final confidence = objectness × class_score.
        """
        boxes       = []
        confidences = []
        class_ids   = []

        for output in outputs:
            for detection in output:
                # First 4 values are box coords, 5th is objectness
                scores     = detection[5:]        # one score per class
                class_id   = np.argmax(scores)    # most likely class
                confidence = scores[class_id] * detection[4]  # final conf

                if confidence < self.conf_thresh:
                    continue

                # Convert normalized coords back to pixel coords
                cx = int(detection[0] * img_w)
                cy = int(detection[1] * img_h)
                w  = int(detection[2] * img_w)
                h  = int(detection[3] * img_h)

                # cv2.dnn gives center x,y — convert to top-left x,y
                x = cx - w // 2
                y = cy - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # ── Non-Max Suppression ───────────────────────────────────────────────
        # Without NMS we'd get multiple overlapping boxes for the same object
        # NMS keeps the highest confidence box and removes others that
        # overlap it by more than nms_thresh IoU
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.conf_thresh,
            self.nms_thresh
        )

        final_boxes       = []
        final_confidences = []
        final_class_ids   = []

        if len(indices) > 0:
            for i in indices.flatten():
                final_boxes.append(boxes[i])
                final_confidences.append(confidences[i])
                final_class_ids.append(class_ids[i])

        return final_boxes, final_confidences, final_class_ids


    def detect(self, image):
        """
        Full detection pipeline on one image.

        Args:
            image : BGR numpy array (as loaded by cv2.imread)

        Returns:
            boxes       : list of [x, y, w, h] in pixel coords
            confidences : list of float confidence scores
            class_ids   : list of int class indices
        """
        img_h, img_w = image.shape[:2]

        # Preprocess
        blob = self.preprocess(image)
        self.net.setInput(blob)

        # Forward pass through all 3 YOLO output layers
        outputs = self.net.forward(self.output_layers)

        # Postprocess — filter + NMS
        boxes, confidences, class_ids = self.postprocess(
            outputs, img_h, img_w
        )

        return boxes, confidences, class_ids


    def get_class_name(self, class_id):
        """Return class name string for a given class id."""
        return self.classes[class_id] if class_id < len(self.classes) else '?'


    def get_color(self, class_id):
        """Return BGR color tuple for a given class id."""
        color = self.colors[class_id]
        return (int(color[0]), int(color[1]), int(color[2]))
