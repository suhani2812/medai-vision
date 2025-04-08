from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login


# Medical information for each condition
MEDICAL_INFO = {
    # Brain conditions
    'glioma': {
        'description': 'Glioma is a type of tumor that occurs in the brain and spinal cord. It begins in the glial cells that surround and support nerve cells.',
        'symptoms': 'Headaches, nausea, vomiting, seizures, memory problems, changes in behavior, difficulty with balance, vision problems, and speech difficulties.',
        'treatment': 'Treatment typically involves surgery to remove as much of the tumor as possible, followed by radiation therapy and chemotherapy.',
        'prognosis': 'The prognosis varies depending on the grade and location of the glioma, but generally higher-grade gliomas have a more serious outlook.'
    },
    'meningioma': {
        'description': 'Meningioma is a tumor that forms on membranes that cover the brain and spinal cord just inside the skull. Most meningiomas are noncancerous (benign).',
        'symptoms': 'Headaches, seizures, blurred vision, weakness in the limbs, and changes in personality or memory.',
        'treatment': 'If the meningioma is small and not causing symptoms, it may be monitored. Otherwise, surgery is the primary treatment, sometimes followed by radiation therapy.',
        'prognosis': 'Meningiomas are often benign and slow-growing. The prognosis is generally good with proper treatment.'
    },
    'notumor': {
        'description': 'No tumor detected in the brain scan. This means the scan does not show evidence of abnormal tissue growth that would indicate a tumor.',
        'symptoms': 'While no tumor was found, your symptoms may be caused by other conditions that require further investigation.',
        'treatment': 'Since no tumor was detected, treatment will depend on identifying the actual cause of any symptoms you may be experiencing.',
        'prognosis': 'The absence of a tumor is generally good news, but further evaluation may be needed to determine the cause of any symptoms.'
    },
    'pituitary': {
        'description': 'A pituitary tumor is an abnormal growth in the pituitary gland, which is located at the base of the brain. Most pituitary tumors are noncancerous (benign).',
        'symptoms': 'Headaches, vision problems, fatigue, unexplained weight changes, increased thirst and urination, mood changes, and hormonal imbalances.',
        'treatment': 'Treatment options include medication to control hormone production, surgery to remove the tumor, and radiation therapy.',
        'prognosis': 'Most pituitary tumors are benign and treatable. The prognosis is generally good with proper treatment.'
    },
    
    # Lung conditions
    'benign': {
        'description': '''A benign lung nodule is a small growth on the lung that is not cancerous. These are quite common and usually don't require treatment.''',
        'symptoms': '''Most benign lung nodules don't cause symptoms. They are often found during imaging tests for other conditions.''',
        'treatment': '''Benign nodules typically don't require treatment, but may be monitored with regular imaging to ensure they don't change over time.''',
        'prognosis': 'The prognosis for benign lung nodules is excellent, as they are not cancerous and rarely cause health problems.'
    },
    'malignant': {
        'description': 'A malignant lung finding indicates lung cancer, which is the uncontrolled growth of abnormal cells in one or both lungs.',
        'symptoms': 'Persistent cough, coughing up blood, chest pain, hoarseness, weight loss, shortness of breath, and recurrent respiratory infections.',
        'treatment': 'Treatment options include surgery, chemotherapy, radiation therapy, targeted drug therapy, and immunotherapy, depending on the type and stage of cancer.',
        'prognosis': 'The prognosis depends on the type and stage of lung cancer, but early detection and treatment generally lead to better outcomes.'
    },
    'normal': {
        'description': 'Your lung scan appears normal, with no evidence of nodules, masses, or other concerning findings.',
        'symptoms': '''While your scan is normal, any symptoms you're experiencing may be caused by other conditions that require further investigation.''',
        'treatment': 'Since no abnormalities were detected, treatment will depend on identifying the actual cause of any symptoms you may be experiencing.',
        'prognosis': 'A normal lung scan is good news, but further evaluation may be needed to determine the cause of any symptoms.'
    }
}

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None:
        print("Loading LLaMA model...")
        # Set model path
        model_id = "meta-llama/Llama-3.2-3B-Instruct"
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load the model with 4-bit quantization for efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True
        )
        print("Model loaded!")

def generate_initial_response(prediction, organ_type):
    load_model()
    
    # Get medical information for the prediction
    info = MEDICAL_INFO.get(prediction, {})
    
    # Create a system prompt with medical information
    system_prompt = f"""You are MedAI Vision's medical consultant, a helpful and compassionate AI doctor explaining {organ_type} scan results to a patient.
    
The scan shows: {prediction}

Medical Information:
- Description: {info.get('description', 'Information not available')}
- Common Symptoms: {info.get('symptoms', 'Information not available')}
- Treatment Options: {info.get('treatment', 'Information not available')}
- Prognosis: {info.get('prognosis', 'Information not available')}

Provide a clear, compassionate explanation of these results to the patient, explaining the condition in simple terms, discussing potential next steps, and offering reassurance where appropriate. Be informative but not alarming.
"""
    
    # Create the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Please explain my scan results to me."}
    ]
    
    # Convert messages to prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs=inputs.input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

def generate_response(user_message, prediction, organ_type):
    load_model()
    
    # Get medical information for the prediction
    info = MEDICAL_INFO.get(prediction, {})
    
    # Create a system prompt with medical information
    system_prompt = f"""You are MedAI Vision's medical consultant, a helpful and compassionate AI doctor explaining {organ_type} scan results to a patient.
    
The scan shows: {prediction}

Medical Information:
- Description: {info.get('description', 'Information not available')}
- Common Symptoms: {info.get('symptoms', 'Information not available')}
- Treatment Options: {info.get('treatment', 'Information not available')}
- Prognosis: {info.get('prognosis', 'Information not available')}

Answer the patient's question clearly and compassionately. Provide accurate medical information in simple terms.
"""
    
    # Create the conversation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Convert messages to prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs=inputs.input_ids,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # For testing
    print("Testing LLaMA model responses...")
    response = generate_initial_response("glioma", "brain")
    print("Response:", response)
