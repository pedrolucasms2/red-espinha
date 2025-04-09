import os   
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras_cv
from tensorflow.keras import *
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import *
from tensorflow.keras.models import load_model
from tqdm.notebook import tqdm
from datetime import datetime

class_mapping = {
    0: "espinha"
}
backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone", include_rescaling=True)
YOLOV8_model = keras_cv.models.YOLOV8Detector(
    num_classes=len(class_mapping),
    bounding_box_format="xyxy",
    backbone=backbone,
    fpn_depth=5
)

YOLOV8_model.load_weights("/Users/pedrolucasmirandasouza/Documents/2024.2/Marketing/faceRec/skin-disease-recognition/yolo_acne_detection.h5")

def visualize_predict_detections(model, dataset, bounding_box_format, confidence_threshold=0.2):
    images, y_true = next(iter(dataset.take(1)))
    y_pred_dict = model.predict(images, verbose=0)
    if "boxes" in y_pred_dict and "confidence" in y_pred_dict and "classes" in y_pred_dict:
        boxes = y_pred_dict["boxes"]
        confidences = y_pred_dict["confidence"]
        classes = y_pred_dict["classes"]
        num_detections = y_pred_dict["num_detections"][0]
        indices = confidences[0] >= confidence_threshold
        filtered_boxes = boxes[0][indices]
        filtered_confidences = confidences[0][indices]
        filtered_classes = classes[0][indices]
        print("Número de detecções antes da filtragem:", num_detections)
        print("Número de detecções após a filtragem:", len(filtered_boxes))
        y_pred = {"boxes": filtered_boxes, "classes": filtered_classes}
        y_pred = tf.nest.map_structure(tf.convert_to_tensor, y_pred)
        y_pred = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), y_pred)
        y_pred = keras_cv.bounding_box.to_ragged(y_pred)
        keras_cv.visualization.plot_bounding_box_gallery(
            images,
            value_range=(0, 255),
            bounding_box_format=bounding_box_format,
            y_true=y_true,
            y_pred=y_pred,
            true_color=(192, 57, 43),
            pred_color=(255, 235, 59),
            scale=8,
            font_scale=0.8,
            line_thickness=2,
            dpi=100,
            rows=1,
            cols=1,
            show=True,
            class_mapping=class_mapping,
        )
    else:
        print("Erro: As chaves 'boxes', 'confidence' ou 'classes' não foram encontradas no dicionário de predições.")
        print("Conteúdo do dicionário de predições:", y_pred_dict)
        return

def create_dataset_from_image(image_path):
    def img_preprocessing(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        return img

    image = img_preprocessing(tf.constant(image_path))
    image = tf.expand_dims(image, axis=0)

    resizing = keras_cv.layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(1.0, 1.0),
        bounding_box_format="xyxy")

    bounding_boxes = {
        "classes": tf.zeros((1, 0), dtype=tf.float32),
        "boxes": tf.zeros((1, 0, 4), dtype=tf.float32)
    }

    dataset = tf.data.Dataset.from_tensors(({"images": image, "bounding_boxes": bounding_boxes}))
    dataset = dataset.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    def dict_to_tuple(inputs):
        return inputs["images"], inputs["bounding_boxes"]
    dataset = dataset.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset

def main():
    print("🔍 Tentando iniciar a câmera...")
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("❌ Não foi possível abrir a câmera.")
        return

    print("✅ Câmera iniciada.")
    print("Pressione 's' para capturar a imagem e rodar o modelo, ou 'q' para sair.")

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("❌ Falha ao capturar frame.")
            break
        else:
            print("🖼️ Frame capturado com sucesso.")

        cv2.imshow("Reconhecimento de Espinhas", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"captura_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            print(f"📸 Imagem capturada: {image_path}")

            try:
                dataset = create_dataset_from_image(image_path)
                visualize_predict_detections(
                    YOLOV8_model,
                    dataset,
                    bounding_box_format="xyxy",
                    confidence_threshold=0.2
                )
            except Exception as e:
                print(f"⚠️ Erro durante predição: {e}")

        elif key == ord('q'):
            print("👋 Encerrando.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()