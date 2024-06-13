"""Main module for the FastAPI application."""
from io import BytesIO
from typing import Union

import numpy as np
from deepface import DeepFace
from fastapi import FastAPI, UploadFile
from PIL import Image

app = FastAPI()


@app.get("/")
def read_root() -> dict:
    """
    Root endpoint of the API.

    Returns a JSON response with a greeting message.
    """
    return {"Hello": "World"}


@app.post("/detect")
def detect_face(file: UploadFile) -> dict:
    """
    Endpoint for detecting faces in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.
    """
    pass


@app.post("/add_face")
def add_face(file: UploadFile) -> str:
    """
    Endpoint for adding a face to the face recognition model.

    Args:
        file (UploadFile): The image file containing the face to be added.
    Returns:
        str: A message indicating the success or failure of the operation.
    """
    pass


@app.post("/encode")
def encode(file: UploadFile) -> list:
    """
    Endpoint for encoding a face in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.
    Returns:
        list: The encoded representation of the face.
    """
    pass
