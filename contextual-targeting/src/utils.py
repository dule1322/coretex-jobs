from zipfile import ZipFile
from pathlib import Path

from coretex import cache
from coretex.folder_management import FolderManager


def fetchModelFile(modelUrl: str, modelName: str, modelSuffix: str) -> Path:
    if not cache.exists(modelUrl):
        cache.storeUrl(modelUrl, modelName)

    modelPath = cache.getPath(modelUrl)
    if modelPath is None:
        raise ValueError

    with ZipFile(modelPath, "r") as zipFile:
        zipFile.extractall(FolderManager.instance().cache)

    return modelPath.with_suffix(modelSuffix)
