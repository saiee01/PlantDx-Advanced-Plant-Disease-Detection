# PlantDx-Advanced-Plant-Disease-Detection
Advanced Plant Disease Detection System using Transfer Learning and Deep Learning Techniques.

PlantCare AI – Detailed Project Workflow

The workflow of PlantCare AI consists of two major parts:

User Interaction Workflow (Application Flow)

Complete Project Development Workflow (ML + Deployment Pipeline)

 1️⃣ User Interaction Workflow (End-to-End System Flow)

This describes how the system works when a user uses the web application.

Step 1: User Interface Interaction

The user opens the web application built using Flask.

The homepage displays:

Project title

Upload button

Instructions for capturing a plant leaf image

The user navigates to the upload section.

Technology Used:

Flask (Backend)

HTML, CSS (Frontend)

Step 2: Image Selection

The user:

Uploads an image from device
OR

Captures an image using mobile camera

The image must clearly show a plant leaf.

Supported formats:

JPG

PNG

JPEG

Step 3: Image Processing (Preprocessing Stage)

Once uploaded:

The image is saved temporarily in the static/ folder.

The image is resized to 224 × 224 pixels.

Image is converted into array format.

Pixel values are normalized using MobileNetV2 preprocessing.

The image is reshaped into batch format (1, 224, 224, 3).

This ensures compatibility with the pre-trained model.

Step 4: Model Prediction

The processed image is passed to the trained MobileNetV2 Transfer Learning Model.

The model outputs:

A probability score for each of the 38 disease classes.

The system selects:

The class with highest probability using argmax().



If class index 2 has highest probability → That disease is predicted.

Step 5: Result Display

The web application then:

Converts predicted index into disease name

Displays:

🌱 Predicted Disease Name

📊 Confidence Score (%)

💡 Basic treatment recommendation

The result is shown on the same page or a result page.
