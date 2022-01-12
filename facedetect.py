import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO

# To install this module, run:
# python -m pip install Pillow
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

# This key will serve all examples in this document.
KEY = "0d90f98d41c44406b45d1255009e4d19"

# This endpoint will be used in all examples in this quickstart.
ENDPOINT = "https://face-rec-dcube.cognitiveservices.azure.com/"

# 1- Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

# 2- Detect a face in an image that contains a single face
single_face_image_url = "https://dcubefr-my.sharepoint.com/:i:/g/personal/hany_akoury_dcube_fr/EceyFaUsRBdEiGdOXPq1m9ABLFB1EOO16EGiy2Mu9xYIuA?e=psAhkI"

single_image_name = os.path.basename(single_face_image_url)
# We use detection model 3 to get better performance.
detected_faces = face_client.face.detect_with_url(
    url=single_face_image_url, detection_model='detection_03')
if not detected_faces:
    raise Exception('No face detected from image {}'.format(single_image_name))

# 3- Convert width height to a point in a rectangle
def getRectangle(faceDictionary):
    rect = faceDictionary.face_rectangle
    left = rect.left
    top = rect.top
    right = left + rect.width
    bottom = top + rect.height

    return ((left, top), (right, bottom))


def drawFaceRectangles():
    # Download the image from the url
    response = requests.get(single_face_image_url)
    img = Image.open(BytesIO(response.content))

    # For each face returned use the face rectangle and draw a red box.
    print('Drawing rectangle around face... see popup for results.')
    draw = ImageDraw.Draw(img)
    for face in detected_faces:
        draw.rectangle(getRectangle(face), outline='red')

# Display the image in the default image browser.
    img.show()


# Uncomment this to show the face rectangles.
drawFaceRectangles()
