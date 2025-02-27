from pathlib import Path

import csv
import logging

from coretex import ImageSample, folder_manager, Model
from PIL import Image, ImageOps

import numpy as np
import torch
import torchvision.transforms as transforms

from .model import CNNModel


def calculateAccuracy(prediction: float, groundtruth: float) -> float:
    sigma = 0.5 / 3.5  # This value was chosen because the width of the Gaussian distribution function graph is the most suitable for the strictness of the metric

    # Calculating the value of the Gaussian normal distribution function, translated so that the peak corresponds to the groundtruth, for the predicted value
    accuracy = float(np.exp(-0.5 * ((prediction - groundtruth) / sigma)**2))

    return accuracy * 100


def run(modelPath: Path, dataset: list[tuple[ImageSample, float]], transform: transforms.Compose) -> tuple[Path, Path, float]:
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(modelPath, map_location = device))
    model.eval()

    sampleResultsPath = folder_manager.temp / "sample_results.csv"
    sampleAccuracies: list[float] = []

    with sampleResultsPath.open("w") as file:
        writer = csv.DictWriter(file, ["id", "name", "prediction", "groundtruth", "accuracy"])
        writer.writeheader()

        for sample, quality in dataset:
            logging.info(f">> [ImageQuality] Validating sample \"{sample.name}\"...")
            image = ImageOps.exif_transpose(Image.open(sample.imagePath).convert("RGB"))
            if image is None:
                raise ValueError(f">> [ImageQuality] Failed to open image {sample.name}")

            if quality == 0:
                logging.warning("\tSample has quality == 0")

            output = model(transform(image).unsqueeze(0)).squeeze(0).item()
            accuracy = calculateAccuracy(output, quality)

            writer.writerow({
                "id": sample.id,
                "name": sample.name,
                "prediction": f"{output:.2f}",
                "groundtruth": quality,
                "accuracy": f"{accuracy:.2f}"
            })

            sampleAccuracies.append(accuracy)

            logging.info(f"\tGroudtruth: {quality}")
            logging.info(f"\tPrediction: {output}")
            logging.info(f"\tAccuracy: {accuracy}")

    datasetResultsCsvPath = folder_manager.createTempFolder("dataset_results") / "dataset_results.csv"
    with open(datasetResultsCsvPath, "w", newline = "") as file:
        writer = csv.DictWriter(file, fieldnames = ["accuracy", "accuracy STD"])
        writer.writeheader()
        writer.writerow({
            "accuracy": f"{sum(sampleAccuracies) / len(sampleAccuracies):.2f}",
            "accuracy STD": f"{np.std(sampleAccuracies):.2f}"
        })

    return sampleResultsPath, datasetResultsCsvPath, sum(sampleAccuracies) / len(sampleAccuracies)
