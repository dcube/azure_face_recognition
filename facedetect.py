import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse


from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

from facefunctions import *

# 0- Explain and show how to create cognitive services API in Azure
# This key will serve all examples in this document.
KEY = ""

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = ""

# 1- Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# 2- Detect a face in an image that contains a single face
single_face_image_url = "https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg"
single_image_name = os.path.basename(single_face_image_url)

detected_faces = face_client.face.detect_with_url(
    url = single_face_image_url, detection_model='detection_03')

if not detected_faces: 
    raise Exception('No face dtected from image {}'.format(single_image_name))

drawFaceRectangles(single_face_image_url,detected_faces)