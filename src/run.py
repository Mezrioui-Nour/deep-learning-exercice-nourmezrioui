import os
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_dataset(image_folder, tsv_file):
    # Lire le fichier TSV pour obtenir les labels
    labels_df = pd.read_csv(tsv_file, sep='\t')
    
    images = []
    labels = []

    # Boucle à travers chaque ligne dans le fichier TSV
    for index, row in labels_df.iterrows():
        image_id = row['Subject']
        label = row['Label']
        
        # Construire le chemin complet de l'image
        image_path = os.path.join(image_folder, image_id + '.png') 

        # Charger l'image
        try:
            image = Image.open(image_path)
            print(type(image))
            print(image)
            images.append(image)
            labels.append(label)
        except IOError:
            print(f"Erreur lors du chargement de l'image {image_path}")

    return images, labels

# Chemins vers vos dossiers d'images et fichiers TSV
train_folder = r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\train' # Remplacez par votre chemin réel
train_tsv = r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\train\subjects.tsv' # Remplacez par votre chemin réel
validation_folder = r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation' # Remplacez par votre chemin réel
validation_tsv = r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\subjects.tsv' # Remplacez par votre chemin réel

def preprocess_images(image_list):
    # Convertir les images en un format compatible avec Keras/TensorFlow
    processed_images = np.array([np.array(image.convert('L')) for image in image_list])  # Convertir en niveaux de gris
    processed_images = processed_images / 255.0  # Normaliser les pixels entre 0 et 1
    processed_images = processed_images.reshape(-1, 121, 121, 1)  # Redimensionner pour le CNN
    return processed_images

def train():
    # Chargement et prétraitement des données d'entraînement
    train_images, train_labels = load_dataset(train_folder, train_tsv)
    train_images_processed = preprocess_images(train_images)

    # Construction du modèle CNN
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(121, 121, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compilation du modèle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    model.fit(train_images_processed, np.array(train_labels), epochs=5, validation_split=0.1)

    return model

def validate(model):
    # Chargement et prétraitement des données de validation
    validation_images, validation_labels = load_dataset(validation_folder, validation_tsv)
    validation_images_processed = preprocess_images(validation_images)

    # Prédiction sur les images de validation
    validation_predictions = model.predict(validation_images_processed)
    validation_predictions = [1 if x > 0.5 else 0 for x in validation_predictions]

    # Générer le fichier output.tsv
    output_df = pd.DataFrame({'Subject': [img.filename.split('/')[-1].split('.')[0] for img in validation_images], 'Label': validation_predictions})
    
    def fct(row):
        
        chaine=row['Subject']
        sous_chaine = "sub"
        indice = chaine.find(sous_chaine)
        new_val = chaine[indice:]
        row['Subject'] = new_val
        return row

    # Appliquer la fonction à chaque ligne
    output_df = output_df.apply(fct, axis=1)

    output_df.to_csv(r'C:\Users\ITO\Desktop\EXERCICE NOUR\basic-deep-learning-nourmezrioui-main\data\validation\output.tsv', index=False, sep="\t")
    
    

def main() -> None:
    train_images, train_labels = load_dataset(train_folder, train_tsv)
    validation_images, validation_labels = load_dataset(validation_folder, validation_tsv)

    model = train()
    validate(model)

if __name__ == "__main__":
    main()
