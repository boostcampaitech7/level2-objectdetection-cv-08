import numpy as np

def compute_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def calculate_map_at_iou(ground_truths, detections, iou_threshold):
    tp = 0
    fp = 0
    fn = 0
    detected_boxes = []

    for gt in ground_truths:
        matched = False
        for det in detections:
            if any(np.array_equal(det, d) for d in detected_boxes):
                continue
            iou = compute_iou(gt['bbox'], det['bbox'])
            if iou >= iou_threshold:
                tp += 1
                matched = True
                detected_boxes.append(det)
                break
        if not matched:
            fn += 1

    fp = len(detections) - len(detected_boxes)
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision, recall

def compute_map_over_iou_ranges(ground_truths, detections):
    iou_ranges = [0.5, 0.75, 0.95]
    precision_list = []
    recall_list = []

    for iou_threshold in iou_ranges:
        precision, recall = calculate_map_at_iou(ground_truths, detections, iou_threshold)
        precision_list.append(precision)
        recall_list.append(recall)
    
    mean_ap = sum(precision_list) / len(precision_list)

    return mean_ap, precision_list, recall_list
