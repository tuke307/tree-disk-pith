import json
from pathlib import Path
import cv2
import shutil
from typing import Union, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from the given file path and converts it from BGR to RGB format.

    Args:
        image_path: Path to the image file.

    Returns:
        The image as a NumPy array in RGB format.

    Raises:
        FileNotFoundError: If the image file does not exist or cannot be read.
    """
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def load_json(filepath: Union[str, Path]) -> Dict:
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        filepath: Path to the JSON file.

    Returns:
        The loaded JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)

    return data


def write_json(dict_to_save: Dict, filepath: Union[str, Path]) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        dict_to_save: Serializable dictionary to save.
        filepath: Path where to save the JSON file.

    Returns:
        None
    """
    filepath = Path(filepath)

    with open(filepath, "w") as f:
        json.dump(dict_to_save, f, indent=4)


def clear_directory(directory: Path) -> None:
    """
    Clears all files and directories in the specified directory.

    Args:
        directory: The directory to clear.

    Returns:
        None

    Raises:
        FileNotFoundError: If the directory does not exist.
        PermissionError: If the program lacks permissions to modify the directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for item in directory.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            logger.error(f"Error deleting {item}: {e}")


def ensure_directory(dir_path: Path, clear: bool = False) -> Path:
    """
    Ensure a directory exists, optionally clearing it first.

    Args:
        dir_path (Path): Directory path
        clear (bool): Whether to clear existing contents

    Returns:
        Path: Resolved directory path
    """
    dir_path = dir_path.resolve()

    if dir_path.exists() and clear:
        clear_directory(dir_path)

    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
