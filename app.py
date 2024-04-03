from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
import os
import io
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, LeakyReLU, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, Add, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from PIL import Image
import pandas as pd
import imutils
from datetime import datetime
from imutils.video import VideoStream

# Constants for file paths
TRAIN_FOLDER = "dataset/train_set"
MODEL_PATH = "models/vgg_model.h5"
FACENET_PATH = "models/facenet_keras.h5"
LABEL_CLASSES_PATH = "dataset/label_encoder_classes.npy"
ATTENDANCE_FILE_PATH = 'dataset/attendance.xlsx'

# Initialize Flask app
app = Flask(__name__)

# Load models and set up necessary objects
trained_model = load_model(MODEL_PATH)
facenet_model = load_model(FACENET_PATH)
detector = MTCNN()
label_encoder = LabelEncoder()
session = {}

def get_face_embeddings(face_images):
    # Function to get face embeddings from images
    embeddings = []
    for face_image in face_images:
        img = tf.image.resize(face_image, (160, 160))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        embedding = facenet_model.predict(img)
        embeddings.append(embedding[0])
    return np.array(embeddings)

# Define the index route
@app.route('/')
def index():
    initialize_attendance()
    return render_template('index.html')


def log_attendance(student_number):
    # Function to log student attendance
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    session['attendance'][student_number] = {'Status': 'Present', 'Time': current_time}
    
# Define the video feed route for streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    # Generator function for video frames and attendance logging
    global vs
    vs = VideoStream(src=0).start()
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(LABEL_CLASSES_PATH)
    attendance_logged = set()
    try:
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = Image.fromarray(frame_rgb)
            faces = detector.detect_faces(np.array(frame_rgb))
            for face in faces:
                x, y, w, h = face['box']
                face_image = frame_rgb.crop((x, y, x+w, y+h)).resize((160, 160))
                face_embedding = get_face_embeddings([face_image])
                predicted_index = np.argmax(trained_model.predict(face_embedding))
                predicted_label = label_encoder.classes_[predicted_index]
                if predicted_label not in attendance_logged:
                    log_attendance(predicted_label)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(predicted_label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        pass

@app.route('/process_images', methods=["POST"])
def process_images():
    # Route to process uploaded images and log attendance
    image_files = request.files.getlist("images")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(LABEL_CLASSES_PATH)
    if not image_files:
        return jsonify({"error": "No images provided!"}), 400
    predictions = []
    attendance_logged = set()
    for image_file in image_files:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        faces = detector.detect_faces(image)
        if not faces:
            predictions.append({"Image Name": image_file.filename, "Predicted Label": "No Face Detected"})
            continue
        face_images = [image[face['box'][1]:face['box'][1] + face['box'][3], face['box'][0]:face['box'][0] + face['box'][2]] for face in faces]
        face_embeddings = get_face_embeddings(face_images)
        # Batch prediction for all embeddings
        predicted_indices = np.argmax(trained_model.predict(face_embeddings), axis=1)
        predicted_labels = label_encoder.classes_[predicted_indices]
        for label in predicted_labels:
            if label not in attendance_logged:
                log_attendance(label)
                attendance_logged.add(label)
            predictions.append({"Image Name": image_file.filename, "Predicted Label": label})
    return download_attendance()

def extract_faces_with_augmentation(image_path, datagen=None):
    # Function to extract faces with data augmentation
    detector = MTCNN()
    face_data = []
    labels = []
    image_name = os.path.basename(image_path)
    print(image_name)
    image = Image.open(image_path)
    faces = detector.detect_faces(np.array(image))
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        face = image.crop((x, y, x+w, y+h)).resize((160, 160))
        if datagen is not None:
            for _ in range(20):
                augmented_faces = datagen.flow(np.array(face).reshape(1, 160, 160, 3), batch_size=1).next()
                for augmented_face in augmented_faces:
                    face_data.append(np.array(augmented_face))
                    labels.append(image_name.split('.')[0])
        else:
            face_data.append(np.array(face))
            labels.append(image_name.split('.')[0])
    return np.array(face_data), labels

def create_vgg_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(2)(x)

    x = Conv1D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=output)
    return model


def train_model(embeddings, labels):
    # Function to train a model
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    print("Fit the train data on VGG model")
    
    # Create and compile the VGG model
    input_shape = (X_train.shape[1], 1)  # Use the shape of 1D input data
    num_classes = len(np.unique(y_train))  
    model = create_vgg_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up callbacks
    checkpoint = ModelCheckpoint("models/vgg_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    lr_scheduler = LearningRateScheduler(lambda epoch: 0.001 * (0.5 ** (epoch // 5)))
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # One-hot encode labels
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_val_encoded = to_categorical(y_val, num_classes=num_classes)

    # Fit the model using train and validation data
    history = model.fit(X_train, y_train_encoded, batch_size=32,
                        epochs=20,
                        validation_data=(X_val, y_val_encoded), 
                        callbacks=[checkpoint, lr_scheduler, early_stopping])
     
@app.route("/register_student", methods=["POST"])
def register_student():
    # Route to register a student and update the model
    student_number = request.form.get("studentNumber")
    student_image = request.files.get("studentImage")
    if not student_number or not student_image:
        return jsonify({"message": "Missing student number or student image"}), 400
    image_path = os.path.join(TRAIN_FOLDER, student_number + ".jpg")
    student_image.save(image_path)
    try:
        existing_embeddings = np.load("dataset/embeddings.npy")
        existing_labels = np.load("dataset/train_labels.npy").tolist()
        label_classes = np.load(LABEL_CLASSES_PATH).tolist()
    except FileNotFoundError:
        existing_embeddings = np.empty((0, 128))
        existing_labels = []
        label_classes = []
    if student_number in label_classes:
        return jsonify({"message": "Student number already exists"}), 400
    datagen = ImageDataGenerator(
        rotation_range=20,width_shift_range=0.2,
        height_shift_range=0.2,shear_range=0.2,
        zoom_range=0.2,horizontal_flip=True,
        brightness_range=[0.75, 1.25],fill_mode='nearest')
    train_images, train_labels = extract_faces_with_augmentation(image_path, datagen)
    existing_labels.extend(train_labels)
    new_embeddings = get_face_embeddings(train_images)
    embeddings = np.concatenate((existing_embeddings, new_embeddings), axis=0)
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(existing_labels)
    num_classes = len(label_encoder.classes_)
    train_model(embeddings, train_labels_encoded)
    np.save("dataset/embeddings.npy", embeddings)
    np.save("dataset/train_labels.npy", np.array(existing_labels))
    np.save(LABEL_CLASSES_PATH, label_encoder.classes_)
    initialize_attendance()
    response_data = {"message": "Student registered successfully"}
    return jsonify(response_data), 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global vs  
    if 'vs' in globals():
        if vs is not None:
            vs.stop()
            del globals()['vs']  # Remove the global reference to the video stream
        
        
        response = jsonify({'status': 'success', 'message': 'Camera stopped successfully'})
    else:
        response = jsonify({'status': 'error', 'message': 'Camera not running'})
    return response

# Function to initialize attendance data
def initialize_attendance():
    # Load all student names from label classes
    all_students = np.load(LABEL_CLASSES_PATH).tolist()
    # Initialize a dictionary with all students marked as absent
    # Each student's data includes their attendance status and the timestamp
    session['attendance'] = {student: {'Status': 'Absent', 'Time': ''} for student in all_students}

# Function to download attendance data as an Excel file
@app.route('/download_attendance', methods=['GET'])
def download_attendance():
    # Convert the session data to a list of dictionaries
    attendance_records = []
    for student_number, attendance_data in session.get('attendance', {}).items():
        attendance_records.append({
            'Student Number': student_number,
            'Status': attendance_data['Status'],
            'Time': attendance_data['Time'] if attendance_data['Time'] else 'N/A'  # Replace empty time with 'N/A'
        })

    # Create a pandas DataFrame
    df = pd.DataFrame(attendance_records)
    
    # Convert the DataFrame to an Excel file
    excel_path = 'attendance.xlsx'  
    df.to_excel(excel_path, index=False)
    # Clear the attendance data in the session
    initialize_attendance()
    
    # Send the file to the user
    return send_file(excel_path, as_attachment=True, download_name="attendance.xlsx")

if __name__ == '__main__':
    app.run(debug=True)
