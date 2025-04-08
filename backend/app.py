from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
import resnet50
import srgan
import cyclegan
import llama

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload and results folders
UPLOAD_FOLDER = '../uploads'
RESULTS_FOLDER = '../results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    organ_type = request.form.get('organType', 'brain')
    image_type = request.form.get('imageType', 'mri')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        
        # Save the uploaded file
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        # Process with ResNet50
        if organ_type == 'brain':
            prediction, confidence = resnet50.predict_image_class(
                resnet50.resnet_brain, 
                ['glioma', 'meningioma', 'notumor', 'pituitary'], 
                input_path
            )
        else:  # lungs
            prediction, confidence = resnet50.predict_image_class(
                resnet50.resnet_lungs, 
                ['benign', 'malignant', 'normal'], 
                input_path
            )
        
        # Process with SRGAN
        sr_filename = f"sr_{unique_filename}"
        sr_output_path = os.path.join(app.config['RESULTS_FOLDER'], sr_filename)
        srgan.super_resolve_image(input_path, sr_output_path)
        
        # Process with CycleGAN
        cg_filename = f"cg_{unique_filename}"
        cg_output_path = os.path.join(app.config['RESULTS_FOLDER'], cg_filename)
        
        # Initialize the translator
        translator = cyclegan.ImageTranslatorGAN()
        
        # Preprocess the image
        original_img = translator.preprocess(input_path)
        
        # Translate the image (opposite of input type)
        translated_img = translator.img2img_translation(original_img, image_type)
        
        # Save the translated image
        translator.save_translated(translated_img, cg_output_path)
        
        # Get LLaMA response based on prediction
        llama_response = llama.generate_initial_response(prediction, organ_type)
        
        # Return all results
        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'organType': organ_type,
            'imageType': image_type,
            'originalImage': f"/api/images/uploads/{unique_filename}",
            'srImage': f"/api/images/results/{sr_filename}",
            'cgImage': f"/api/images/results/{cg_filename}",
            'llamaResponse': llama_response
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    prediction = data.get('prediction', '')
    organ_type = data.get('organType', 'brain')
    
    response = llama.generate_response(user_message, prediction, organ_type)
    
    return jsonify({'response': response})

@app.route('/api/images/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/images/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
