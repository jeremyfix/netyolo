# coding: utf-8

# External imports
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO

# Global variables
app = Flask(__name__)

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")


@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream)

    results = model(image)

    print(results.probs, results.names)

    # Convert processed image to base64 string
    # buffered = io.BytesIO()
    # image.save(buffered, format="JPEG")
    # img_str = base64.b64encode(buffered.getvalue()).decode()

    # Create a response
    response_data = {
        "message": "Image processed successfully",
        "label": "hello",
    }

    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
