"""Flask 推理 API：图片检测 + 大图切片推理"""
import io
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

MODEL = None
MODEL_PATH = None
TILING_ENABLED = False
TILE_SIZE = 640
OVERLAP = 0.2


def load_model(model_path: str):
    global MODEL, MODEL_PATH
    MODEL = YOLO(model_path)
    MODEL_PATH = model_path
    print(f"[INFO] Model loaded from {model_path}")


def tile_detect(image: Image.Image, conf_thres: float = 0.25, iou_thres: float = 0.45):
    """大图切片检测：将图片切片后逐块推理，NMS 合并结果"""
    img_w, img_h = image.size
    stride = int(TILE_SIZE * (1 - OVERLAP))

    all_boxes = []
    all_scores = []
    all_classes = []

    y_positions = sorted(set(
        list(range(0, max(1, img_h - TILE_SIZE), stride)) + [max(0, img_h - TILE_SIZE)]
    ))
    x_positions = sorted(set(
        list(range(0, max(1, img_w - TILE_SIZE), stride)) + [max(0, img_w - TILE_SIZE)]
    ))

    for ty in y_positions:
        for tx in x_positions:
            tile = image.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))
            results = MODEL(tile, conf=conf_thres, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                all_boxes.append([x1 + tx, y1 + ty, x2 + tx, y2 + ty])
                all_scores.append(conf)
                all_classes.append(cls)

    if not all_boxes:
        return []

    boxes_array = np.array(all_boxes, dtype=np.float32)

    import torch
    nms_input = torch.cat([
        torch.from_numpy(boxes_array),
        torch.from_numpy(np.array(all_scores, dtype=np.float32)).unsqueeze(1),
    ], dim=1)
    keep = torch.ops.torchvision.nms(nms_input, iou_thres)

    results_list = []
    for idx in keep:
        i = int(idx.item())
        results_list.append({
            "x1": int(boxes_array[i][0]),
            "y1": int(boxes_array[i][1]),
            "x2": int(boxes_array[i][2]),
            "y2": int(boxes_array[i][3]),
            "class_id": int(all_classes[i]),
            "class_name": MODEL.names.get(int(all_classes[i]), f"class_{all_classes[i]}"),
            "confidence": round(float(all_scores[i]), 4),
        })

    return results_list


@app.route("/detect", methods=["POST"])
def detect():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    if TILING_ENABLED and max(image.size) > TILE_SIZE:
        detections = tile_detect(image)
    else:
        results = MODEL(image, conf=0.25, verbose=False)
        boxes = results[0].boxes
        detections = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                    "class_id": int(box.cls[0]),
                    "class_name": MODEL.names.get(int(box.cls[0]), "unknown"),
                    "confidence": round(float(box.conf[0]), 4),
                })

    return jsonify({"detections": detections, "count": len(detections)})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


def main():
    parser = argparse.ArgumentParser(description="Flask inference API for YOLOv8")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--tiling", action="store_true")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    global TILING_ENABLED, TILE_SIZE, OVERLAP
    TILING_ENABLED = args.tiling
    TILE_SIZE = args.tile_size
    OVERLAP = args.overlap

    load_model(args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
