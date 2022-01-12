import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse


# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

from facefunctions import *

# This key will serve all examples in this document.
KEY = "0d90f98d41c44406b45d1255009e4d19"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://face-rec-dcube.cognitiveservices.azure.com/"

# 1- Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# 2- Detect a face in an image that contains a single face
single_face_image_url = "https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg"

single_image_name = os.path.basename(single_face_image_url)
# We use detection model 3 to get better performance.
detected_faces = face_client.face.detect_with_url(
    url=single_face_image_url, detection_model='detection_03')
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# Uncomment this to show the face rectangles.
drawFaceRectangles(single_face_image_url,detected_faces)
