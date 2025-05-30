from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os

# Parameters
model = load_model("model_fold_2.h5")
np.random.seed(42)
n_iterations, samples_per_class = 10, 50

labels = ['no', 'yes'] # yes: green electro parking spot # 'no' = 0, 'yes' = 1
img_size = 250  # Zielgrösse Bilder        
def get_data(data_dir):
    X = []  # Bilddaten
    y = []  # zugehörigen Labels 0 oder 1 
    
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)

        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]    # Bild einlesen, in RGB umwandeln 
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Bild auf einheitliche Grösse skalieren
                X.append(resized_arr)
                y.append(class_num)
            except Exception as e:
                print(f"Fehler bei {img}: {e}")

    return np.array(X), np.array(y)
x_data, y_data = get_data('data')
x_data = x_data / 255.0
y_data = y_data.astype('float32')

for i in range(n_iterations):
    # Indices of each class
    idx_yes = np.where(y_data == 1)[0]
    idx_no = np.where(y_data == 0)[0]

    # Randomly select 50 from each class
    selected_yes = np.random.choice(idx_yes, size=samples_per_class, replace=False)
    selected_no = np.random.choice(idx_no, size=samples_per_class, replace=False)
    selected_indices = np.concatenate([selected_yes, selected_no])
    np.random.shuffle(selected_indices)  # shuffle to mix classes

    # Get images and labels
    x_sample, y_sample = x_data[selected_indices], y_data[selected_indices]
    # Predict
    predictions = model.predict(x_sample)
    pred_labels = (predictions > 0.5).astype("int32").reshape(-1)
    # Report
    print(f"\n🔁 Run {i+1}")
    print(classification_report(y_sample.astype(int), pred_labels, target_names=['kein E-Parkplatz (0)', 'grüner E-Parkplatz (1)']))

    # Confusion Matrix
    cm = confusion_matrix(y_sample.astype(int), pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['kein E-Parkplatz', 'grüner E-Parkplatz'],
                yticklabels=['kein E-Parkplatz', 'grüner E-Parkplatz'])
    plt.xlabel('Vorhergesagte Klasse')
    plt.ylabel('Tatsächliche Klasse')
    plt.title(f'Confusion Matrix – Run {i+1}')
    plt.show()
