import tensorflow as tf
import MySQLdb
import mysql.connector
import csv
import random
from mysql.connector import Error
from mysql.connector import errorcode
from datetime import datetime
from api import servicelNotification
# Object detection imports
from utils import backbone
#import datetime
import cv2
import numpy as np
from utils import visualization_utils as vis_util

def targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_object, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('the_output.avi', fourcc, fps, (width, height))
        #today= datetime.date.today()
        today=datetime.now()
        now= datetime.now()
        cur_time= now.strftime("%H:%M:%S")
        day=datetime.today().strftime("%A")
        device_id=1
        rand_id=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61]
		# input video
        cap = cv2.VideoCapture(input_video)

        
        with detection_graph.as_default():
          with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            
            #******************************************* DATABASE CONNECTION*****************************
            
            db = MySQLdb.connect("localhost","root","","python_test" )
            cursor = db.cursor()

            #**********************************************************************************************
            while(cap.isOpened()):
                ret, frame = cap.read()                

                if not  ret:
                    print("end of the video file...")
                    break
                
                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Visualization of the results of a detection.        
                counter, csv_line, the_result = vis_util.visualize_boxes_and_labels_on_image_array(cap.get(1),
                                                                                                      input_frame,
                                                                                                      1,
                                                                                                      is_color_recognition_enabled,
                                                                                                      np.squeeze(boxes),
                                                                                                      np.squeeze(classes).astype(np.int32),
                                                                                                      np.squeeze(scores),
                                                                                                      category_index,
                                                                                                      targeted_objects=targeted_object,
                                                                                                      use_normalized_coordinates=True,
                                                                                                      line_thickness=4)
               
                if(len(the_result) == 0):
                    cv2.putText(input_frame, "...", (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX) 
                    print ("no people")
                else:
                    cv2.putText(input_frame, the_result, (10, 35), font, 0.8, (0,255,255),2,cv2.FONT_HERSHEY_SIMPLEX)
                    print("no of people" +str(the_result))
                    random_id= random.choice(rand_id)
                    
       #*********************************************DATABASE OPERATION***********************************             
                    
                    data_table= (""" INSERT INTO `crowd_count_1`
                          (`ID`, `Person`, `Date`, `Time`, `Day`, `Device_ID`) VALUES (%s, %s, %s, %s, %s, %s)""", (random_id, the_result, today, cur_time, day, device_id))
                    cursor.execute(*data_table)
                
               # data_table=""" INSERT INTO `demo` VALUES (%s)""", the_result
                
                #data_table= """ INSERT INTO `crowd_count`
                 #         (`number_of_people`, `date`, `time`) VALUES ('6','10/12/93','10:55')"""
                
                #cursor.execute("INSERT INTO demo VALUES (%s)" %(the_result))
                    
                
                    db.commit()
                    
                    sql = "INSERT INTO `final_table` SELECT crowd_count_1.Person AS Crowd_Count, crowd_count_1.Date AS Date, crowd_count_1.Time AS Time, crowd_count_1.Day AS Day, crowd_count_1.Device_ID AS Device_ID, branch_details.State AS State, branch_details.City AS City, branch_details.Location AS Location, branch_details.Pincode AS Pincode FROM crowd_count_1 INNER JOIN branch_details ON crowd_count_1.ID = branch_details.ID "
                    cursor.execute(sql)
                    db.commit()
                    
        #*************************************************************************************************************        
               
                cv2.imshow('object counting',input_frame)
                output_movie.write(input_frame)
                print ("writing frame")
                

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                     

            cap.release()
            cv2.destroyAllWindows()
			

			
input_video = "Bank.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('faster_rcnn_inception_v2_coco_2018_01_28')

#object_counting_api.object_counting(input_video, detection_graph, category_index, 0) # for counting all the objects, disabled color prediction

#object_counting_api.object_counting(input_video, detection_graph, category_index, 1) # for counting all the objects, enabled color prediction


targeted_objects = "person" # (for counting targeted objects) change it with your targeted objects
fps = 25 # change it with your input video fps
width = 1280 # change it with your input video width
height = 720 # change it with your input video height
is_color_recognition_enabled = 0

targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting

#object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects