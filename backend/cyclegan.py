from skimage import exposure
import numpy as np
import cv2
import os

class ImageTranslatorGAN:
    
    ct_gan = r'C:\Users\aryan\Documents\Programs\Projects\medai-vision-webapp\models\cyclegan-ct.keras'
    mri_gan = r'C:\Users\aryan\Documents\Programs\Projects\medai-vision-webapp\models\cyclegan-mri.keras'
    
    def __init__(self):
        self.img_size = (256, 256)
        
    def preprocess(self, image_path):
        # image preprocessing
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        return img

    def img2img_translation(self, image, input_type):
        # img2img translation using simple image processing
        if input_type == 'ct':
            # convert to MRI
            enhanced = exposure.equalize_adapthist(image, clip_limit=0.03)
            return (enhanced * 255).astype(np.uint8)
        elif input_type == 'mri':
            # mri to ct
            adjusted = cv2.convertScaleAbs(image, alpha=0.6, beta=30)
            return cv2.GaussianBlur(adjusted, (5,5), 0)
        else:
            raise ValueError("Invalid input type. Use 'ct' or 'mri'")

    def save_translated(self, translated, output_path):
        cv2.imwrite(output_path, translated)

if __name__ == "__main__":
    # For testing
    input_path = "../dataset/cyclegan/mri/mri-image-2.jpg"
    output_path = "../trans1.png"
    input_type = 'mri'  # 'ct' or 'mri'
    
    #initialize GAN
    translator = ImageTranslatorGAN()

    # preprocessing
    original_img = translator.preprocess(input_path)
    print('done')

    translated_img = translator.img2img_translation(original_img, input_type)
    print('done')

    translator.save_translated(translated_img, output_path)
