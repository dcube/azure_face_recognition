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


# Used in the Person Group Operations and Delete Person Group examples.
PERSON_GROUP_ID = str(uuid.uuid4()) # assign a random ID (or name it anything)

# Used for the Delete Person Group example.
TARGET_PERSON_GROUP_ID = str(uuid.uuid4()) # assign a random ID (or name it anything)


'''
Create the PersonGroup
'''
# Create empty Person Group.
print('Person group:', PERSON_GROUP_ID)
face_client.person_group.create(person_group_id=PERSON_GROUP_ID, name=PERSON_GROUP_ID)

# Define arthur
arthur = face_client.person_group_person.create(PERSON_GROUP_ID, "Arthur")
print('Arthur\'s id: {}'.format(arthur.person_id))

'''
Detect faces and register to correct person
'''
# Find all jpeg images of friends in working directory
arthur_images = [file for file in glob.glob('images/*.jpg') if 'mask' not in file]
#print(' '.join(arthur_images))
# Add to a arthur person
for image in arthur_images:
    w = open(image, 'r+b')
    face_client.person_group_person.add_face_from_stream(PERSON_GROUP_ID, arthur.person_id, w)

'''
Train PersonGroup
'''
print()
print('Training the person group:')
# Train the person group
face_client.person_group.train(PERSON_GROUP_ID)

while (True):
    training_status = face_client.person_group.get_training_status(PERSON_GROUP_ID)
    print("Training status: {}.".format(training_status.status))
    if (training_status.status is TrainingStatusType.succeeded):
        break
    elif (training_status.status is TrainingStatusType.failed):
        face_client.person_group.delete(person_group_id=PERSON_GROUP_ID)
        sys.exit('Training the person group has failed.')
    time.sleep(5)


'''
Identify a face against a defined PersonGroup
'''
# Group image for testing against

test_image_array = glob.glob('images/eval/eval_1.jpg')
image = open(test_image_array[0], 'r+b')


# Detect faces
face_ids = []
# We use detection model 3 to get better performance.
faces = face_client.face.detect_with_stream(image, detection_model='detection_03')
for face in faces:
    face_ids.append(face.face_id)
 

# Identify faces
results = face_client.face.identify(face_ids, PERSON_GROUP_ID)
print('\nIdentifying faces in {}...'.format(os.path.basename(image.name)))
if not results:
    print('No person identified in the person group for faces from {}.'.format(os.path.basename(image.name)))
print ('I have detected {} potential faces\n'.format(len(results)))
for person in results:
    if len(person.candidates) > 0:
        print('Person for face ID {} in {} looks like Arthur with a confidence of {}.'.format(person.face_id, os.path.basename(image.name), person.candidates[0].confidence)) # Get topmost confidence score
          
    else:
        print('No person - identified for face ID {} in {}.'.format(person.face_id, os.path.basename(image.name)))