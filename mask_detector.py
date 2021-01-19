from tensorflow.keras.models import load_model
detector = load_model(r'detection.model')
import tensorflow as tf
import cv2
import numpy


cap = cv2.VideoCapture(0) 


classifier = cv2.CascadeClassifier(r"classifier/haarcascade_frontalface_default.xml")

while True:
    (success, frame) = cap.read()   
    new_image = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0] // 1)) 
    face = classifier.detectMultiScale(new_image) 
    for x,y,w,h in face:
        try:
            face_img = new_image[y:x+h, x:x+w] 
            resized= cv2.resize(face_img,(224,224)) 
            image_array = tf.keras.preprocessing.image.img_to_array(resized)  
            image_array = tf.expand_dims(image_array,0) 
            predictions = detector.predict(image_array) 
            score = tf.nn.softmax(predictions[0])  
            label = numpy.argmax(score)
        except Exception as e:
            print('bad frame')
            
        if label == 0:
            cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(new_image,"mask",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0), 2)
        elif label == 1:
            cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(new_image,'no_mask',(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255), 2)
        else:
            None
    
    cv2.imshow('Detection_window', new_image)
    # print(numpy.argmax(score), 100*numpy.max(score))
    
    key = cv2.waitKey(10) 
    print(numpy.argmax(score), 100*numpy.max(score))
    if key == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()