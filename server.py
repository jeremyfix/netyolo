# coding: utf-8

# External imports
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

# Global variables
app = Flask(__name__)

# Load a pretrained YOLO11n model
model = YOLO("yolo11n-cls.pt")


@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream)

    results = model(image)

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


if __name__ == "__main__":
    app.run(debug=False, port=5000)
