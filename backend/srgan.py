from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Define model path relative to the current file
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
# generator = load_model(os.path.join(model_dir, 'srgan-generator.keras'))
generator = load_model(r'C:\Users\aryan\Documents\Programs\Projects\medai-vision-webapp\models\srgan-generator.keras')

# Parameters for the model
img_width_lr = 64
img_height_lr = 64

def super_resolve_image(input_path, output_path):
    # Load and preprocess the image
    try:
        img = Image.open(input_path)
        img = img.convert('RGB')  # Ensure RGB mode
        
        # Store original dimensions
        original_width, original_height = img.size
        
        # Resize to the input size expected by the generator
        img_lr = img.resize((img_width_lr, img_height_lr), Image.BICUBIC)
        
        # Convert to numpy array and normalize
        img_lr_array = np.array(img_lr) / 255.0 
        
        # Add batch dimension
        img_lr_batch = np.expand_dims(img_lr_array, axis=0)
        
        # Generate the super-resolved image
        print("Generating super-resolution image...")
        img_sr_batch = generator.predict(img_lr_batch)
        
        # Convert back to image format
        img_sr_array = np.clip(img_sr_batch[0] * 255.0, 0, 255).astype(np.uint8)
        img_sr = Image.fromarray(img_sr_array)
        
        # Save the output image
        img_sr.save(output_path)
        
        print(f"Super-resolved image saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return False

if __name__ == "__main__":
    # For testing
    input_path = "../dataset/resnet/lungs/malignant/lungs_malignant_10.jpg"
    output_path = "../image.png"
    
    # Process the image
    super_resolve_image(input_path, output_path)
