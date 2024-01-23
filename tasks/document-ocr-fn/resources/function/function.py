from typing import Any
from pathlib import Path

import time

from coretex import functions

from src import detect_document
from src.model import loadSegmentationModel, loadDetectionModel
from src.image_segmentation import processMask, segmentImage, segmentDetections
from src.ocr import performOCR
from src.object_detection import runObjectDetection


def current() -> int:
    return int(time.time() * 1000)


DOCUMENT_NOT_FOUND = {
    "code": 500,
    "body": {
        "error": "Failed to find document on the provided image"
    }
}

modelsDir = Path.cwd().parent

start = current()
segmentationModel = loadSegmentationModel(modelsDir / "segmentationModel")
print(f"[loadSegmentationModel] {current() - start}")

start = current()
objDetModelWeights = loadDetectionModel(modelsDir / "detectionModel")
print(f"[loadDetectionModel] {current() - start}")


def response(requestData: dict[str, Any]) -> dict[str, Any]:
    imagePath = requestData.get("image")
    if not isinstance(imagePath, Path):
        return functions.badRequest("Input image is invalid")

    start = current()
    predictedMask = detect_document.run(segmentationModel, imagePath)
    print(f"[detect_document.run] {current() - start}")

    start = current()
    predictedMask = processMask(predictedMask)
    print(f">> [processMask] {current() - start}")

    if predictedMask is None:
        return DOCUMENT_NOT_FOUND

    start = current()
    segmentedImage = segmentImage(imagePath, predictedMask)
    print(f">> [segmentImage] {current() - start}")

    if segmentedImage is None:
        return DOCUMENT_NOT_FOUND

    start = current()
    bboxes, classes = runObjectDetection(segmentedImage, objDetModelWeights)
    print(f">> [runObjectDetection] {current() - start}")

    start = current()
    segmentedDetections = segmentDetections(segmentedImage, bboxes)
    print(f">> [segmentDetections] {current() - start}")

    start = current()
    result = performOCR(segmentedDetections, classes)
    print(f">> [performOCR] {current() - start}")

    return functions.success(result)


if __name__ == "__main__":
    print(response({
        "image": Path.home().joinpath("Downloads", "MicrosoftTeams-image (4).png")
    }))
