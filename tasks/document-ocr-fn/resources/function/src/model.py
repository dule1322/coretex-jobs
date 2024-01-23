from pathlib import Path

from ultralytics import YOLO

import tensorflow as tf


def loadSegmentationModel(modelDir: Path) -> tf.lite.Interpreter:
    modelPath = modelDir / "model.tflite"

    interpreter = tf.lite.Interpreter(str(modelPath))
    interpreter.allocate_tensors()

    return interpreter


def loadDetectionModel(modelPath) -> YOLO:
    return YOLO(modelPath / "best.pt")
