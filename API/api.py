from flask import Flask, send_file, jsonify
import cv2
from datetime import datetime
from main import create_dataset_from_image, visualize_predict_detections, YOLOV8_model

app = Flask(__name__)

@app.route("/captura", methods=["GET"])
def capturar():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return jsonify({"erro": "Não foi possível acessar a câmera."}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"erro": "Falha ao capturar frame."}), 500

    image_path = "captura.jpg"
    cv2.imwrite(image_path, frame)

    try:
        dataset = create_dataset_from_image(image_path)
        visualize_predict_detections(YOLOV8_model, dataset, bounding_box_format="xyxy")
    except Exception as e:
        return jsonify({"erro": str(e)}), 500

    return send_file(image_path, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)