import cv2
import os
import dlib
import numpy as np


shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")


cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de charger l'image à partir de {image_path}. Vérifiez le chemin.")
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
    for (x, y, w, h) in faces:
        face_region_gray = gray_image[y:y + h, x:x + w]
        rect = dlib.rectangle(0, 0, face_region_gray.shape[1], face_region_gray.shape[0])
        shape = shape_predictor(face_region_gray, rect)
        face_region_rgb = cv2.cvtColor(face_region_gray, cv2.COLOR_GRAY2RGB)
        face_descriptor = face_recognizer.compute_face_descriptor(face_region_rgb, shape)
        face_encodings.append(np.array(face_descriptor))
    if not face_encodings:
        print(f"Aucun encodage de visage trouvé pour {image_path}")
    return face_encodings

def compare_faces(known_encodings, candidate_encodings, tolerance=0.6):
    for known_encoding in known_encodings:
        for candidate_encoding in candidate_encodings:
            distance = np.linalg.norm(known_encoding - candidate_encoding)
            if distance < tolerance:
                return True
    return False


def find_matching_images(reference_image_path, database_folder, results_folder):
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Dossier {results_folder} créé.")

    reference_encodings = get_face_encodings(reference_image_path)
    if not reference_encodings:
        print(f"Aucun encodage de visage trouvé pour l'image de référence {reference_image_path}.")
        return []

    matching_images = []
    for filename in os.listdir(database_folder):
        image_path = os.path.join(database_folder, filename)
        print(f"Traitement de l'image : {image_path}")

        image_encodings = get_face_encodings(image_path)
        if not image_encodings:
            print(f"Aucun encodage de visage trouvé pour l'image {image_path}.")
            continue

        if compare_faces(reference_encodings, image_encodings):
            matching_images.append(image_path)

            image = load_image(image_path)
            cv2.imshow("Image à sauvegarder", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Copier l'image correspondante dans le dossier des résultats
            result_image_path = os.path.join(results_folder, filename)
            image = load_image(image_path)

            success = cv2.imwrite(result_image_path, image)
            if success:
                print(f"Image correspondante sauvegardée : {result_image_path}")
            else:
                print(f"Erreur lors de la sauvegarde de l'image : {result_image_path}")

    if not matching_images:
        print("Aucune image correspondante trouvée.")
    else:
        print(f"Images correspondantes trouvées et sauvegardées dans le dossier '{results_folder}'.")

    return matching_images



def main():
    reference_image_path = "selena.jpeg"
    database_folder = "faces"
    results_folder = "test_results"

    matching_images = find_matching_images(reference_image_path, database_folder, results_folder)
    if matching_images:
        print("Images correspondantes trouvées et sauvegardées dans le dossier 'résultats'.")
    else:
        print("Aucune image correspondante trouvée dans la base de données.")

if __name__ == "__main__":
    main()
