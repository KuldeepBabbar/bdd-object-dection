import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from tqdm import tqdm

IOU_THRESH = 0.5

# Define object detection classes
CLASSES = [
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck",
]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Load BDD100K-style JSONs
with open("bdd100k_labels_images_val.json") as f:
    gt_data = json.load(f)
with open("faster_rcnn_prediction.json") as f:
    pred_data = json.load(f)

# Step 1: Create dicts from ground truth and prediction
gt_boxes = defaultdict(list)
for item in gt_data:
    image_id = item["name"]
    for label in item.get("labels", []):
        if "box2d" in label and label["category"] in CLASS_TO_IDX:
            box = label["box2d"]
            bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
            cls = CLASS_TO_IDX[label["category"]]
            gt_boxes[image_id].append(
                {
                    "bbox": bbox,
                    "class": cls,
                    "used": False,
                }
            )

# Updated: Parse predictions from your new format
pred_boxes = defaultdict(list)


for frame in pred_data["frames"]:
    image_id = frame["name"]
    for label in frame.get("labels", []):
        if "box2d" in label:
            category = label["category"]
            # Map to standard BDD class names
            if category == "pedestrian":
                category = "person"
            elif category not in CLASS_TO_IDX:
                continue  # skip unknown classes
            cls = CLASS_TO_IDX[category]

            box = label["box2d"]
            bbox = [box["x1"], box["y1"], box["x2"], box["y2"]]
            score = label.get("score", 1.0)  # should always be present

            pred_boxes[image_id].append(
                {
                    "bbox": bbox,
                    "class": cls,
                    "score": score,
                }
            )


# Step 2: IoU calculation
def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
    return iou


# Step 3: Accumulate detections per class
all_scores = defaultdict(list)
all_tp = defaultdict(list)
all_fp = defaultdict(list)
n_gt = defaultdict(int)

for cls_id in range(len(CLASSES)):
    for image_id in gt_boxes:
        gt_cls = [obj for obj in gt_boxes[image_id] if obj["class"] == cls_id]
        preds = [obj for obj in pred_boxes.get(image_id, []) if obj["class"] == cls_id]

        n_gt[cls_id] += len(gt_cls)

        # sort preds by score descending
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        used = np.zeros(len(gt_cls))

        for pred in preds:
            ious = [compute_iou(pred["bbox"], gt["bbox"]) for gt in gt_cls]
            max_iou_idx = np.argmax(ious) if ious else -1
            max_iou = ious[max_iou_idx] if ious else 0

            if max_iou >= IOU_THRESH and used[max_iou_idx] == 0:
                all_tp[cls_id].append(1)
                all_fp[cls_id].append(0)
                used[max_iou_idx] = 1
            else:
                all_tp[cls_id].append(0)
                all_fp[cls_id].append(1)
            all_scores[cls_id].append(pred["score"])

# Step 4: Calculate precision, recall, AP
aps = {}
plt.figure(figsize=(15, 10))

# Step 4: Calculate precision, recall, AP and plot separately
aps = {}
for cls_id, cls_name in enumerate(CLASSES):
    if not all_tp[cls_id]:
        aps[cls_name] = 0.0
        continue

    scores = np.array(all_scores[cls_id])
    tps = np.array(all_tp[cls_id])
    fps = np.array(all_fp[cls_id])

    sorted_indices = np.argsort(-scores)
    tps = tps[sorted_indices]
    fps = fps[sorted_indices]

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)

    precision = cum_tp / (cum_tp + cum_fp + 1e-6)
    recall = cum_tp / (n_gt[cls_id] + 1e-6)

    ap = auc(recall, precision)
    aps[cls_name] = ap

    # Plot separate PR curve
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP={ap:.4f}", color="blue")
    plt.title(f"PR Curve - {cls_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pr_curve_{cls_name.replace(' ', '_')}.png")
    print(f"pr_curve_{cls_name.replace(' ', '_')}.png")

    # plt.show()
    plt.close()
