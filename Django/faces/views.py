from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage
import os
import cv2
import numpy as np

from surveillanceBot.settings import BASE_DIR
 # Create your views here.

def data_generator(self):
    # print(BASE_DIR)
    # url = staticfiles_storage.url('model_data/deploy.prototxt')
    # staticfiles_storage.open('model_data/deploy.prototxt')
    # prototxt_path = staticfiles_storage.url('/modeldata/deploy.prototxt')
    caffemodel_file = staticfiles_storage.open('weights.caffemodel')
    # caffemodel_file = staticfiles_storage.open(caffemodel_path)
    newcontents = caffemodel_file.read()
    # prototxt_file = staticfiles_storage.open('deploy.prototxt')
    # contents = prototxt_file.read()
    print(newcontents)
    # caffemodel_file = staticfiles_storage.open('/modeldata/weights.caffemodel')
    # print(prototxt_path)
    # print(caffemodel_path)
    # base_dir = os.path.dirname(__file__)
    # prototxt_path = os.path.join(BASE_DIR + '/'+'faces' + '/' + 'model_data' + '/' + 'deploy.prototxt')
    # caffemodel_path = os.path.join(BASE_DIR + '/'+'faces' + '/' + 'model_data' + '/' + 'weights.caffemodel')
    # prototxt_path = os.path.join(BASE_DIR , "/faces/model_data/deploy.prototxt")
    # caffemodel_path = os.path.join(BASE_DIR, "/faces/model_data/weights.caffemodel")
    
    # print(prototxt_path,"hello")
    # print(caffemodel_path,"hello")
    # # Read the model
    # model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # # Create directory 'updated_images' if it does not exist
    # if not os.path.exists('updated_images'):
    #     print("New directory created")
    #     os.makedirs('updated_images')

    # # Loop through all images and save images with marked faces
    # for file in os.listdir(staticfiles_storage.url):
    #     file_name, file_extension = os.path.splitext(file)
    #     if (file_extension in ['.png','.jpg']):
    #         print("Image path: {}".format(staticfiles_storage.url('images/') + file))

    #         image = cv2.imread(staticfiles_storage.url('images/') + file)
    #         # print(image)
    #         (h, w) = image.shape[:2]
    #         blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    #         model.setInput(blob)
    #         detections = model.forward()

    #         # Create frame around face
    #         for i in range(0, detections.shape[2]):
    #             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #             (startX, startY, endX, endY) = box.astype("int")

    #             confidence = detections[0, 0, i, 2]

    #             # If confidence > 0.5, show box around face
    #             if (confidence > 0.5):
    #                 cv2.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

    #         # cv2.imwrite(base_dir + '/updated_images/' + file, image)
    #         print("Image " + file + " converted successfully")


def face_extractor(self):
    base_dir = os.path.dirname(__file__)
    prototxt_path = os.path.join(base_dir + '/model_data/deploy.prototxt')
    caffemodel_path = os.path.join(base_dir + '/model_data/weights.caffemodel')

    # Read the model
    model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    # Create directory 'faces' if it does not exist
    if not os.path.exists('faces'):
        print("New directory created")
        os.makedirs('faces')

    # Loop through all images and strip out faces
    count = 0
    for file in os.listdir(base_dir + '/images'):
        file_name, file_extension = os.path.splitext(file)
        if (file_extension in ['.png','.jpg']):
            image = cv2.imread(base_dir + '/images/' + file)

            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            model.setInput(blob)
            detections = model.forward()

            # Identify each face
            for i in range(0, detections.shape[2]):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                confidence = detections[0, 0, i, 2]

                # If confidence > 0.5, save it as a separate file
                if (confidence > 0.5):
                    count += 1
                    frame = image[startY:endY, startX:endX]
                    cv2.imwrite(base_dir + '/faces/' + str(i) + '_' + file, frame)

    print("Extracted " + str(count) + " faces from all images")


def home(request):
    url = staticfiles_storage.url('images/faces.jpg')
    print(url)
    data_generator(url)
    return render(request, "home.html")