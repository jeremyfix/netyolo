# coding: utf-8

# Standard imports

# External imports
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO


class Server:

    def __init__(self):
        self.classification_model = None
        self.detection_model = None
        self.app = Flask(__name__)
        self.setup_routes()

    def setup_routes(self):
        self.app.add_url_rule(
            "/classify", endpoint="classify", view_func=self.classify, methods=["POST"]
        )
        self.app.add_url_rule(
            "/detect", endpoint="detect", view_func=self.detect, methods=["POST"]
        )

    def run(self):
        self.app.run(debug=False, port=5000)

    def classify(self):
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        if self.classification_model is None:
            self.classification_model = YOLO("yolo11n-cls.pt")

        image_file = request.files["image"]
        image = Image.open(image_file.stream)

        results = self.classification_model(image)

        # Only get the results for the first image, we have only one
        results = results[0]
        probs = results.probs
        names = results.names
        top1 = probs.top1
        top1_conf = probs.top1conf.tolist()
        top5 = probs.top5
        top5_conf = probs.top5conf.tolist()

        # with open("classes.txt", "w") as f:
        #     for id_cls, n in names.items():
        #         f.write(f"{id_cls}: {n}\n")

        # Create a response
        response_data = {
            "top1": names[top1],
            "top1conf": top1_conf,
            "top5": [names[i] for i in top5],
            "top5conf": top5_conf,
            "names": names,
        }

        return jsonify(response_data), 200

    def detect(self):
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        if self.detection_model is None:
            self.detection_model = YOLO("yolo11n.pt")

        image_file = request.files["image"]
        image = Image.open(image_file.stream)

        results = self.detection_model(image)

        # Only get the results for the first image, we have only one
        results = results[0]
        boxes = results.boxes
        names = results.names
        conf_thr = 0.5
        boxes = boxes[boxes.conf > conf_thr]

        # Create a response
        response_data = {
            "boxes": boxes.xywhn.tolist(),
            "names": [names[i] for i in boxes.cls.tolist()],
            "scores": boxes.conf.tolist(),
        }

        return jsonify(response_data), 200


if __name__ == "__main__":
    server = Server()
    server.run()
