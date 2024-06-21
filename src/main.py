"""Main module for the FastAPI application."""
import os
import tempfile
import uuid
from io import BytesIO
from typing import Union

import numpy as np
from deepface import DeepFace
from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

app = FastAPI()

backend = "mtcnn"

client = QdrantClient(url="http://localhost:6333")

# client.create_collection(
#     collection_name="faces_collection",
#     vectors_config=VectorParams(size=4096, distance=Distance.DOT)
# )


@app.get("/")
def read_root() -> dict:
    """
    Root endpoint of the API.

    Returns a JSON response with a greeting message.
    """
    return {"Hello": "World"}


def delete_temp_file(file_name: str) -> None:
    """
    Delete tmp file.

    Args:
        file_name (str): Path to the file that needs to be deleted.

    Returns:
        None
    """
    os.unlink(file_name)


def img_encode(file: UploadFile, background_tasks: BackgroundTasks) -> list:
    """Image encode - converts image to vector.

    Args:
        file (UploadFile): _description_
        background_tasks (BackgroundTasks): _description_

    Returns:
        list: _description_
    """
    content = file.file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(content)
    e_o = DeepFace.represent(img_path=tmp.name, detector_backend="dlib")
    background_tasks.add_task(delete_temp_file, tmp.name)
    return e_o[0]["embedding"]


@app.post("/detect")
def detect_face(file: UploadFile, background_tasks: BackgroundTasks) -> list:
    """
    Endpoint for detecting faces in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.
    """
    content = file.file.read()
    # img_bytes = np.fromstring(content, np.uint8)
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(content)
    dfs = DeepFace.extract_faces(img_path=tmp.name, detector_backend=backend)
    background_tasks.add_task(delete_temp_file, tmp.name)
    return dfs[0]["face"].tolist()


@app.post("/add_face")
def add_face(file: UploadFile, background_tasks: BackgroundTasks, name: str = Form(...)) -> str:
    """
    Endpoint for adding a face to the face recognition model.

    Args:
        file (UploadFile): The image file containing the face to be added.
    Returns:
        str: A message indicating the success or failure of the operation.
    """
    vec = img_encode(file, background_tasks)
    client.upsert(
        collection_name="faces_collection",
        wait=True,
        points=[PointStruct(id=str(uuid.uuid4()), vector=vec, payload={"name": name})],
    )

    return "succes"


@app.post("/encode")
def encode(file: UploadFile, background_tasks: BackgroundTasks) -> list:
    """
    Endpoint for encoding a face in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.
    Returns:
        list: The encoded representation of the face.
    """
    return img_encode(file, background_tasks)


@app.post("/recognize")
def recognize(file: UploadFile, background_tasks: BackgroundTasks) -> dict:
    """
    Endpoint for detecting faces in an uploaded image.

    Args:
        file (UploadFile): The uploaded image file.
    """
    vec = encode(file=file, background_tasks=background_tasks)

    search_result = client.search(collection_name="faces_collection", query_vector=vec, limit=1)
    return search_result[0].payload
