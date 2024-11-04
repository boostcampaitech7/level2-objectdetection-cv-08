from pycocotools.cocoeval import COCOeval
import torch
from tqdm import tqdm

def evaluate_fn(model, val_data_loader, coco, device):
    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in tqdm(val_data_loader):
            images = [image.to(device) for image in images]
            outputs = model(images)
            coco_results.extend(parse_outputs(outputs, targets))

    coco_dt = coco.loadRes(coco_results)
    coco_eval = COCOeval(coco, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats[0]  # mAP 0.5:0.95

def parse_outputs(outputs, targets):
    results = []
    for i, output in enumerate(outputs):
        boxes = output['boxes'].cpu().numpy()
        scores = output['scores'].cpu().numpy()
        labels = output['labels'].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            results.append({
                'image_id': int(targets[i]['image_id']),
                'category_id': int(label) - 1,
                'bbox': box.tolist(),
                'score': float(score)
            })
    return results
