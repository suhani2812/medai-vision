from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

# Define model paths relative to the current file
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Load the saved models
# resnet_brain = load_model(os.path.join(model_dir, 'resnet50_brain.keras'))
# resnet_lungs = load_model(os.path.join(model_dir, 'resnet50_lungs.keras'))
resnet_brain = load_model(r'C:\Users\aryan\Documents\Programs\Projects\medai-vision-webapp\models\resnet50_brain.keras')
resnet_lungs = load_model(r'C:\Users\aryan\Documents\Programs\Projects\medai-vision-webapp\models\resnet50_lungs.keras')

# Function to preprocess a single image for prediction
def preprocess_single_image(image_path):
    # Read and preprocess the image
    img_height, img_width = 224, 224
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Function to predict the class of an image
def predict_image_class(model, classes, image_path):
    processed_image = preprocess_single_image(image_path)
    predictions = model.predict(processed_image)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    
    return classes[class_idx], confidence

if __name__ == "__main__":
    # For testing
    image_path = "../dataset/resnet/lungs/benign/lungs_bengin_53.jpg"
    
    input_type = 'lungs'  # 'brain' or 'lungs'
    
    if input_type == 'brain':
        brain_classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        predicted_class, confidence = predict_image_class(resnet_brain, brain_classes, image_path)
        print(f"Prediction: {predicted_class} with confidence: {confidence:.2f}")
    
    elif input_type == 'lungs':
        lungs_classes = ['benign', 'malignant', 'normal']
        predicted_class, confidence = predict_image_class(resnet_lungs, lungs_classes, image_path)
        print(f"Prediction: {predicted_class} with confidence: {confidence:.2f}")
    
    else:
        print("Invalid input type. Use 'brain' or 'lungs'")
