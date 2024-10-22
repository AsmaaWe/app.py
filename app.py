import os
import cv2
import dlib
import numpy as np
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


upload_folder = "uploads"
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)


# Initialisation des modèles de reconnaissance faciale
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
frontal_cascade_path = os.path.join(os.getcwd(), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(frontal_cascade_path)


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image à partir de {image_path}.")
    return image


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def get_face_encodings(image_path):
    image = load_image(image_path)
    faces = detect_faces(image)
    face_encodings = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if len(faces) == 0:

        return None  # Aucun visage détecté

    for (x, y, w, h) in faces:
        face_region_gray = gray_image[y:y + h, x:x + w]
        rect = dlib.rectangle(0, 0, face_region_gray.shape[1], face_region_gray.shape[0])
        shape = shape_predictor(face_region_gray, rect)
        face_region_rgb = cv2.cvtColor(face_region_gray, cv2.COLOR_GRAY2RGB)
        face_descriptor = face_recognizer.compute_face_descriptor(face_region_rgb, shape)
        face_encodings.append(np.array(face_descriptor))

    return face_encodings if face_encodings else None


def compare_faces(known_encodings, candidate_encodings, tolerance=0.6):
    for known_encoding in known_encodings:
        for candidate_encoding in candidate_encodings:
            distance = np.linalg.norm(known_encoding - candidate_encoding)
            if distance < tolerance:
                return True
    return False


def find_matching_images(face_image_path, database_folder, results_folder, non_encoded_folder):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    if not os.path.exists(non_encoded_folder):
        os.makedirs(non_encoded_folder)

    reference_encodings_face = get_face_encodings(face_image_path)

    if reference_encodings_face is None:
        return [], "Aucun visage encodé trouvé pour l'image de référence.", non_encoded_folder

    matching_images = []
    non_encoded_images = []

    for filename in os.listdir(database_folder):
        image_path = os.path.join(database_folder, filename)
        image_encodings_face = get_face_encodings(image_path)

        if image_encodings_face is None:
            # Déplacez l'image non encodée dans le dossier non_encoded
            non_encoded_image_path = os.path.join(non_encoded_folder, filename)
            image = load_image(image_path)
            cv2.imwrite(non_encoded_image_path, image)
            non_encoded_images.append(non_encoded_image_path)
            continue

        # Comparaison des visages encodés
        if image_encodings_face and compare_faces(reference_encodings_face, image_encodings_face):
            matching_images.append(image_path)
            result_image_path = os.path.join(results_folder, filename)
            image = load_image(image_path)
            cv2.imwrite(result_image_path, image)

    if not matching_images:
        return [], "Aucune image correspondante trouvée.", non_encoded_images
    return matching_images, "Images correspondantes trouvées et sauvegardées.", non_encoded_images


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_images():
    if 'face_image' not in request.files or 'database_folder' not in request.form or 'results_folder_path' not in request.form or 'results_folder_name' not in request.form:
        return redirect(url_for('index'))

    face_image = request.files['face_image']
    database_folder = request.form['database_folder']
    results_folder_path = request.form['results_folder_path']  # Chemin du dossier
    results_folder_name = request.form['results_folder_name']  # Nom du dossier

    # Dossier pour les images non encodées
    non_encoded_folder = os.path.join(results_folder_path, 'non_encoded')

    # Créer le chemin complet du dossier de résultats
    results_folder = os.path.join(results_folder_path, results_folder_name)

    # Créer le dossier de résultats si nécessaire
    os.makedirs(results_folder, exist_ok=True)

    face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], face_image.filename)
    face_image.save(face_image_path)
    print(f"Fichier de visage sauvegardé à : {face_image_path}")

    matching_images, message, non_encoded_images = find_matching_images(face_image_path, database_folder,
                                                                        results_folder, non_encoded_folder)

    return render_template('result.html', matching_images=matching_images, non_encoded_images=non_encoded_images,
                           message=message)


if __name__ == "__main__":
    app.run(debug=True)

