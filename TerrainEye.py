import tensorflow as tf
import cv2
import os
import numpy as np
import time

model_path = r'C:\Users\harvi\Codebases\TerrainEye\TerrainEye.h5'
source_dir = r'C:\Users\harvi\Codebases\TerrainEye\dataset_split\train'
model = tf.keras.models.load_model(model_path,custom_objects=None, compile=True)
print("model loaded successfully")

def preprocess_image(img):
    img = cv2.resize(img, (256, 256))
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def get_class_name(pred_index):
    class_names = []
    for class_name in os.listdir(source_dir):
        class_names.append(class_name)
    print("The classes are: ", class_names)
    return class_names[pred_index]
 
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()
PREDICTION_INTERVAL = 10

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Camera Feed', frame)
    current_time = time.time()

    if current_time - last_prediction_time >= PREDICTION_INTERVAL:
        processed_img = preprocess_image(frame)
        prediction = model.predict(processed_img)
        pred_class = np.argmax(prediction[0])
        confidence = prediction[0][pred_class]

        class_name = get_class_name(pred_class)
        print(f"\n\n\nPredicted: {class_name} \n(Confidence: {confidence:.2f})")
        
        # Update last prediction time
        last_prediction_time = current_time

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()