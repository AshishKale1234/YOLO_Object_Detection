
import cv2
import numpy as np
from pathlib import Path


def draw_detections(image, boxes, confidences, class_ids,
                    classes, colors, show_conf=True):
    """
    Draws bounding boxes, class labels and confidence scores
    on the image using OpenCV.

    Args:
        image      : BGR numpy array
        boxes      : list of [x, y, w, h]
        confidences: list of float scores
        class_ids  : list of int class indices
        classes    : list of class name strings
        colors     : numpy array of BGR colors per class
        show_conf  : whether to show confidence score on label

    Returns:
        output : annotated BGR image (copy of input)
    """
    output = image.copy()

    for box, conf, cls_id in zip(boxes, confidences, class_ids):
        x, y, w, h = box
        color = tuple(int(c) for c in colors[cls_id])
        label = f"{classes[cls_id]} {conf:.2f}" if show_conf \
                else classes[cls_id]

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        # ── Label background ──────────────────────────────────────────────────
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 1
        (tw, th), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Fill background rectangle behind text
        cv2.rectangle(output,
                      (x, y - th - baseline - 4),
                      (x + tw + 4, y),
                      color, -1)

        # ── Label text ────────────────────────────────────────────────────────
        cv2.putText(output, label,
                    (x + 2, y - baseline - 2),
                    font, font_scale,
                    (255, 255, 255), thickness,
                    cv2.LINE_AA)

    # ── Detection count ───────────────────────────────────────────────────────
    count_text = f"{len(boxes)} object(s) detected"
    cv2.putText(output, count_text,
                (10, output.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, count_text,
                (10, output.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 1, cv2.LINE_AA)

    return output


def save_result(image, output_path):
    """Save annotated image to disk."""
    cv2.imwrite(str(output_path), image)


def show_in_notebook(image, title='', figsize=(12, 8)):
    """Display BGR image inline in Colab notebook."""
    import matplotlib.pyplot as plt
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title(title, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def make_comparison_grid(original, annotated, title=''):
    """
    Stack original and annotated images side by side.
    Both images must be same height — resize annotated if needed.
    """
    if original.shape[0] != annotated.shape[0]:
        annotated = cv2.resize(
            annotated,
            (int(annotated.shape[1] * original.shape[0] / annotated.shape[0]),
             original.shape[0])
        )

    # Add column headers
    orig_labeled = original.copy()
    ann_labeled  = annotated.copy()

    for img, label in [(orig_labeled, 'Original'),
                       (ann_labeled,  'Detections')]:
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return np.hstack([orig_labeled, ann_labeled])
