import time, math, argparse, cv2, sys, torch
import numpy as np
import json
import random
import os
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sample:

    def __init__(self, args):

        self.args = args
        # classes for the age and gender category
        self.ageList = ['(8-16)', '(17-22)', '(23-27)', '(28-32)', '(33-37)', '(38-43)', '(48-53)', '(60-100)']
        self.ages = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(21-24)", "(25-32)",
                     "(33-37)", "(38-43)", "(44-47)", "(48-53)", "(54-59)", "(60-100)"]
        self.genders = ["Male", "Female"]

        # loading face detector pretrained model
        faceProto = "../models/face_detector/opencv_face_detector.pbtxt"
        faceModel = "../models/face_detector/opencv_face_detector_uint8.pb"
        self.faceNet = cv2.dnn.readNet(faceModel, faceProto)

        # age detector pretrained model
        ageProto = "../models/age_detector/age_deploy.prototxt"
        ageModel = "../models/age_detector/age_net.caffemodel"
        self.ageNet = cv2.dnn.readNet(ageModel, ageProto)

        # gender detector pretrained model
        genderProto = "../models/gender_detector/gender_deploy.prototxt"
        genderModel = "../models/gender_detector/gender_net.caffemodel"
        self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

        # model mean values to subtract from facenet model
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    @staticmethod
    def getFaceBox(net, frame, conf_threshold=0.7):
        frameOpencvDnn = frame.copy()  
        frameHeight = frameOpencvDnn.shape[0]  
        frameWidth = frameOpencvDnn.shape[1] 

        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)  
        detections = net.forward() 
        bboxes = []  
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2] 

           
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)  # top left x co-ordinate
                y1 = int(detections[0, 0, i, 4] * frameHeight)  # top left y co-ordinate
                x2 = int(detections[0, 0, i, 5] * frameWidth)  # bottom right x co-ordinate
                y2 = int(detections[0, 0, i, 6] * frameHeight)  # bottom right y co-ordinate
                bboxes.append([x1, y1, x2, y2])  # append the co-ordinates list computed above in the bounding box list

                
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)

        return frameOpencvDnn, bboxes

    def caffeInference(self):

        img_folder =self.args.output + "Image_for_swap"

        os.makedirs(img_folder, exist_ok=True)

        Json_Input = self.args.Json_Input + "Input.json"

        data = []


        for filename in os.listdir(self.args.input):
            if filename.endswith((".mp4", ".mov")): 
                video_path = os.path.join(self.args.input, filename)




            m_count = 0
            fm_count = 0

        
            cap = cv2.VideoCapture(video_path if video_path else 0)
            padding = 20 

        
            while True:
                
                
                hasFrame, frame = cap.read()  

                
                if not hasFrame:
                    # cv2.waitKey()
                    break

            
                frameFace, bboxes = self.getFaceBox(self.faceNet, frame)

            
                if not bboxes:
                    print("No face Detected, Checking next frame")
                    cv2.putText(frameFace, "NO FACE DETECTED!", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                cv2.LINE_AA)  
                    # cv2.imshow("Age Gender Demo", frameFace)  # display empty frames with message
                else:
                
                    for bbox in bboxes:
                    
                        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                    
                        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                        
                        self.genderNet.setInput(blob) 
                        genderPreds = self.genderNet.forward()  
                        gender = self.genders[genderPreds[0].argmax()]  

                    
                        # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

                        if gender == "Male":
                            m_count = m_count + 1

                        if gender == "Female":
                            fm_count = fm_count + 1


           
                    
                        self.ageNet.setInput(blob) 
                        agePreds = self.ageNet.forward()  
                        age = self.ageList[agePreds[0].argmax()] 

                        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))




            if m_count > fm_count:
                l_gender = "Male"

                video_capture = cv2.VideoCapture(video_path)
                image_saved = False  

                video_name = os.path.splitext(os.path.basename(video_path))[0]

                frame_number = 0

                while video_capture.isOpened() and not image_saved:
                    ret, frame = video_capture.read()

                    if not ret:
                        break 
                
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        face_image = frame [(y-20):y + (h+20), (x-20):x + (w+20)]

                        
                        roi_gray = gray[y:y + h, x:x + w]
                        eyes = eye_cascade.detectMultiScale(roi_gray)

                        for (ex, ey, ew, eh) in eyes:
                            
                            # cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

                            if len(eyes) == 2 :

                                face_image_save = cv2.resize(frame, (640, 480))
                                # face_image_save = cv2.resize(face_image, (640, 480))

                                frame_name = f'{video_name}__frame--{frame_number}.jpg'
                                cv2.imwrite(os.path.join(img_folder, frame_name), face_image_save)

                                item = {
                                    "Image":frame_name,
                                    "Gender":l_gender,
                                }
                                data.append(item)
                                
                                
                                image_saved = True 

                                break

                    frame_number += 1

                video_capture.release()
            if fm_count > m_count:

                l_gender = "Female"


                video_capture = cv2.VideoCapture(video_path)
                image_saved = False  

                video_name = os.path.splitext(os.path.basename(video_path))[0]

                frame_number = 0

                while video_capture.isOpened() and not image_saved:
                    ret, frame = video_capture.read()

                    if not ret:
                        break 
                
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        
                        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        face_image = frame [(y-20):y + (h+20), (x-20):x + (w+20)]

                        
                        roi_gray = gray[y:y + h, x:x + w]
                        eyes = eye_cascade.detectMultiScale(roi_gray)

                        for (ex, ey, ew, eh) in eyes:
                            
                            # cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 0, 255), 2)

                            if len(eyes) == 2 :

                                face_image_save = cv2.resize(frame, (640, 480))
                                # face_image_save = cv2.resize(face_image, (640, 480))

                                frame_name = f'{video_name}__frame--{frame_number}.jpg'
                                cv2.imwrite(os.path.join(img_folder, frame_name), face_image_save)
                                
                                item = {
                                    "Image": frame_name,
                                    "Gender":l_gender,
                                }
                                data.append(item)

                                image_saved = True 

                                break

                    frame_number += 1

                video_capture.release()


            print("Gender--> " ,l_gender )


        with open(Json_Input, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        print("JSON file 'output.json' has been created.")
        
    def extract_image_path(self,image_path):
    
        image_path = image_path.split('__frame')[0]

        return image_path

    def extract_video_path(self,video_path):
        
        base_name = os.path.basename(video_path)
        return os.path.splitext(base_name)[0]
    
    def jsoncreation(self):
        print("jsoncreation")

        Json_Output = self.args.Json_output + "Output.json"

        Jdata = []
        swap_i = 1

        for filename in os.listdir(self.args.input):
            if filename.endswith((".mp4", ".mov")): 
                video_path = os.path.join(self.args.input, filename)

            male_count = 0
            fmale_count = 0

        
            cap = cv2.VideoCapture(video_path if video_path else 0)
            padding = 20 

        
            while True:
                
                
                hasFrame, frame = cap.read()  

                
                if not hasFrame:
                    break

            
                frameFace, bboxes = self.getFaceBox(self.faceNet, frame)

            
                if not bboxes:
                    print("No face Detected, Checking next frame")
                    cv2.putText(frameFace, "NO FACE DETECTED!", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                                cv2.LINE_AA)  
                    
                else:
                
                    for bbox in bboxes:
                    
                        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                            max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

                    
                        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

                        
                        self.genderNet.setInput(blob) 
                        genderPreds = self.genderNet.forward()  
                        gender = self.genders[genderPreds[0].argmax()]  

                    
                        
                        if gender == "Male":
                            male_count = male_count + 1

                        if gender == "Female":
                            fmale_count = fmale_count + 1

                        self.ageNet.setInput(blob) 
                        agePreds = self.ageNet.forward()  
                        age = self.ageList[agePreds[0].argmax()] 




            if male_count > fmale_count:

                l_gender = "Male"


                with open(self.args.Json_output, 'r') as file:
                    data = json.load(file)

                matching_images = [item['Image'] for item in data if item['Gender'] == l_gender]


                if not matching_images:
                    return "No matching images found for the specified gender."

                # Check if the video_path matches any of the image paths
                
                random.shuffle(matching_images)
            
                for image_path in matching_images:
                    if self.extract_image_path(image_path) != self.extract_video_path(video_path):
                        item_1 = {
                                    "Video" : filename,
                                    "Image": image_path,
                                    "Swapped_video": f"swapped_video_{swap_i}.mp4",
                                    "Gender":l_gender,
                                    "Age":age
                        }
                        Jdata.append(item_1)
                        swap_i = swap_i + 1
                        break
                    
                    
                        

                
            if fmale_count > male_count:

                l_gender = "Female"


                with open(self.args.Json_output, 'r') as file:
                    data = json.load(file)

                matching_images = [item['Image'] for item in data if item['Gender'] == l_gender]


                if not matching_images:
                    return "No matching images found for the specified gender."

                # Check if the video_path matches any of the image paths
                
                random.shuffle(matching_images)
            
                for image_path in matching_images:
                    if self.extract_image_path(image_path) != self.extract_video_path(video_path):
                        item_1 = {
                                    "reference_video" : filename,
                                    "Refernece_img": image_path,
                                    "output_video": f"swapped_video_{swap_i}.mp4",
                                    "Gender":l_gender,
                                    "Age":age
                        }
                        Jdata.append(item_1)
                        swap_i = swap_i + 1
                        break


        with open(Json_Output, 'w') as json_file:
            json.dump(Jdata, json_file, indent=4)

        print("JSON file has been created.")




        
parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the cropped  image.')

parser.add_argument('-ij', '--Json_Input', type=str, default="",
                    help='Path to Input Json path.')

parser.add_argument('-oj', '--Json_output', type=str, default="",
                    help='Path to output Json path.')

args = parser.parse_args()
s = Sample(args)
s.caffeInference()
s.jsoncreation()

