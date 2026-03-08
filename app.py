import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

app = Flask(__name__)
app.secret_key = 'plantcare_secret_key'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload directory if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the Keras model
print("Loading model...")
model = tf.keras.models.load_model('models/best_plant_model.keras')
print("Model loaded successfully!")

# Plant disease class labels (38 classes based on common plant disease datasets)
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Standard input size for MobileNetV2
    img_array = np.array(img)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype('float32')
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_disease(image_path):
    """Predict disease from image"""
    preprocessed = preprocess_image(image_path)
    predictions = model.predict(preprocessed)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    
    class_name = CLASS_NAMES[predicted_class]
    
    # Parse the class name to get plant type and condition
    parts = class_name.split('___')
    plant_type = parts[0].replace('_', ' ')
    condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
    
    is_healthy = 'healthy' in condition.lower()
    
    return {
        'plant_type': plant_type,
        'condition': condition,
        'confidence': confidence,
        'is_healthy': is_healthy,
        'class_name': class_name
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(url_for('upload'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('upload'))
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(os.times().elapsed * 1000))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        try:
            result = predict_disease(filepath)
            
            # Store result in session
            session['prediction'] = result
            session['image_path'] = f"uploads/{unique_filename}"
            
            # Redirect to appropriate result page
            if result['is_healthy']:
                return redirect(url_for('result_healthy'))
            else:
                return redirect(url_for('result_disease'))
                
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('upload'))
    else:
        flash('Invalid file format. Please upload JPG, JPEG, or PNG.')
        return redirect(url_for('upload'))

@app.route('/result/healthy')
def result_healthy():
    prediction = session.get('prediction', {})
    image_path = session.get('image_path', '')
    
    # Default values if no prediction in session
    if not prediction:
        return redirect(url_for('upload'))
    
    return render_template('result-healthy.html', 
                           prediction=prediction,
                           image_path=image_path)

@app.route('/result/disease')
def result_disease():
    prediction = session.get('prediction', {})
    image_path = session.get('image_path', '')
    
    # Default values if no prediction in session
    if not prediction:
        return redirect(url_for('upload'))
    
    return render_template('result-disease.html', 
                           prediction=prediction,
                           image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)

