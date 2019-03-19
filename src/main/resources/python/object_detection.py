print("Importing Libraries...")
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import json
import datetime
import pymongo
import pandas as pd

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from pymongo import MongoClient
from pprint import pprint
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from utils import label_map_util

from utils import visualization_utils as vis_util

print("\nImports completed!")

MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

print("\nChecking/ Downloading model..")
if not os.path.exists(MODEL_NAME + '/frozen_inference_graph.pb'):
	print ('Downloading the model..')
	opener = urllib.request.URLopener()
	opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
	tar_file = tarfile.open(MODEL_FILE)
	for file in tar_file.getmembers():
	  file_name = os.path.basename(file.name)
	  if 'frozen_inference_graph.pb' in file_name:
	    tar_file.extract(file, os.getcwd())
	print ('Download complete!')
else:
	print ('\nModel already exists!')
    
print("\nLoading Tensorflow model...")
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
print("\nLoading successful!")

print("\nLoading Label Map..")
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)
print("\nCompleted!")

import cv2
cap = cv2.VideoCapture(0)
#cap1 = cv2.VideoCapture("bonfire.mov")
print("\nLoading Video..")

with open('hitlist.json', 'w') as f:
    print('', file=f)
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   ret = True
   
   while (ret):
      ret,image_np = cap.read()
      #ret,image_np = cap1.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
    
     # Here output the category as string and score to terminal
      threshold = 0.5
     
      
      
      for index, value in enumerate(classes[0]):
          if scores[0, index] > threshold:
              obj = category_index.get(value)
              with open('hitlist.json', 'w') as f:
                print('{"Records":[', file=f)
              with open('hitlist.json', 'a') as g:
                 currentDT = datetime.datetime.now() 
                 print(json.dumps(obj), file=g)
                  
              with open('hitlist.json', 'a') as h:
                    print("]}", file=h)   
                 
              
            
              print("Loading data into Mongodb")
              import json
              import pymongo

              connection = pymongo.MongoClient("mongodb://localhost", 27017)
              db=connection.mongo_demo
              record = db.risha10
              page = open("hitlist.json", 'r')
              parsed = json.loads(page.read())

              for item in parsed["Records"]:
                record.insert(item) 
              db.risha10.update_many({"Date": {"$exists": False}}, {"$set": {"Date": datetime.datetime.now()}})
              print("\nCompleted!", currentDT )
      
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      cv2.imshow('image',cv2.resize(image_np,(1280,960)))
      
    
    
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
      