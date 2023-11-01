import time, math, argparse, cv2, sys, torch
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sample:

    def __init__(self, args):

        self.args = args
        # classes for the age and gender category
        self.ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
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

        with open(self.args.input, 'r') as json_file:
            data = json.load(json_file)

        num_sets = len(data)

        print(f'There are {num_sets} sets in the JSON file.')

        New_Jdata = []
  
        for entry in data:
            source_path = entry.get('Refernece_img', '')
            target_path = entry.get('reference_video', '')
            output_path = entry.get('output_video', '')

        
            cap = cv2.VideoCapture(target_path if target_path else 0)
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

                        
                        self.genderNet.setInput(
                            blob) 
                        genderPreds = self.genderNet.forward()  
                        gender = self.genders[genderPreds[0].argmax()]  

                    
                        # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
                    

                    
                        self.ageNet.setInput(blob) 
                        agePreds = self.ageNet.forward()  
                        age = self.ageList[agePreds[0].argmax()] 

                        # print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
                        

            print("Gender--> " ,gender )
            print("Age--->   ", age)
            item = {
                "output_video": output_path,
                "Refernece_img": source_path,
                "reference_video": target_path,
                "Gender":gender,
                "Age":age
            }
            New_Jdata.append(item)

        with open(self.args.output, 'w') as json_file:
            json.dump(New_Jdata, json_file, indent=4)

        
parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('-i', '--input', type=str,
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument('-o', '--output', type=str, default="",
                    help='Path to output the prediction in case of single image.')

args = parser.parse_args()
s = Sample(args)
s.caffeInference()

