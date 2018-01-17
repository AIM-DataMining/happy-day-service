# Test the Emotion Detection System
import cv2
import logging
import sys
from EmotionRecognition import EmotionRecognition
from Constants import *
import numpy as np
from sklearn.metrics import classification_report
from tensorflow import keras as k

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        # determine whether the image is color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Image read from buffer
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)

    faces = cascade_classifier.detectMultiScale(image,scaleFactor = 1.3, minNeighbors = 5)

    if not len(faces) > 0:
        return None

    # initialize the first face as having maximum area, then find the one with max_area
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face
    face = max_area_face

    # extract ROI of face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]

    try:
        # resize the image so that it can be passed to the neural network
        image = cv2.resize(image, (SIZE_FACE,SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("----->Problem during resize")
        return None

    return image

def testloop(filename, model_number):
    # Initialize object of EMR class
    network = EmotionRecognition()
    network.build_network(model_number)
    #Load the model from the file
    network.load_model(filename)

    cap = cv2.VideoCapture(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    feelings_faces = []

    # append the list with the emoji images
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread(EMOJIS_FOLDER + emotion + '.png', -1))

    while True:
        #Read a image from the stream
        ret, frame = cap.read()

        #Find the face in the image an cut it out
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, 1.3, 5)

        #Compute the proabilities for emotions
        result = network.predict(format_image(frame))

        if result is not None:
            #Write the different emotions with a bar shown the proability
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1);
                cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[0][index] * 100), (index + 1) * 20 + 4),
                              (255, 0, 0), -1)

            #Find the emotion with the highest proability and write a Emoji in the video stream
            maxindex = np.argmax(result[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, EMOTIONS[maxindex], (10, 360), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
            face_image = feelings_faces[maxindex]

            for c in range(0, 3):
                # the shape of face_image is (x,y,4)
                # the fourth channel is 0 or 1
                # in most cases it is 0, so, we assign the roi to the emoji
                # you could also do:
                # frame[200:320,10:130,c] = frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
                frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)

        if not len(faces) > 0:
            # do nothing if no face is detected
            a = 1
        else:
            # draw box around face with maximum area
            max_area_face = faces[0]
            for face in faces:
                if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
                    max_area_face = face
            face = max_area_face
            (x, y, w, h) = max_area_face
            frame = cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

        cv2.imshow('Video', cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def testAccuracy(filename, model_number):

    # Initialize object of EMR class
    network = EmotionRecognition()
    network.build_network(model_number)
    # Load the model from the file
    network.load_model(filename)

    # eval_data_gen = k.preprocessing.image.ImageDataGenerator(
    #     rescale=1. / 255.
    # )
    # eval_gen = eval_data_gen.flow_from_directory(
    #     "data/test",
    #     target_size=None,
    #     batch_size=1,
    #     class_mode="categorical",
    #     color_mode="grayscale"
    # )
    train_set, validation_set = network.generate_test_data()
    metrics = network.model.evaluate_generator(validation_set, steps=10)

    result = network.model.predict_generator(validation_set, steps=1)

    val = validation_set.classes
    res_class = k.np_utils.probas_to_classes(result)

    print(classification_report(val, res_class))
    print(network.model.metrics_names)

    return {"model": "test",
            "metrics": metrics,
            "names": network.model.metrics_names
            }


    # y_test = []
    # files = glob.glob("D:/workspace/Data-Mining Projekt/Live/data/test/*.png")
    # for myFile in files:
    #     print(myFile)
    #     image = cv2.imread(myFile)
    #     y_test.append(image)
    #
    # y_test = np.argmax(validation_set, axis=1)  # Convert one-hot to index
    #
    # y_predict = network.model.predict(y_test)
    #
    # print(classification_report(Y_test, y_pred))

#Entry of the application
if __name__ == "__main__":
  if len(sys.argv) <= 1:
    exit(0)

  if sys.argv[1] == 'train':
    network = EmotionRecognition()
    network.full_training(sys.argv[3])
    network.save_model(sys.argv[2])
    print('Training finished and model saved')
  elif sys.argv[1] == 'testloop':
      testloop(sys.argv[2], sys.argv[3])
  elif sys.argv[1] == 'test':
      logging.info(testAccuracy(sys.argv[2], sys.argv[3]))
