import os
import numpy as np
import shutil
from sklearn.cluster import DBSCAN, AffinityPropagation
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Konfiguracja
DATASET_PATH = r"C:\Users\antek\PycharmProjects\PythonProject\Final_images_dataset\Final_images_dataset"
RESULTS_PATH = r"C:\Users\antek\PycharmProjects\PythonProject\categorized_images"
MODEL_NAME = "google/vit-base-patch16-224-in21k"

# Modele
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
vit_processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
vit_model = ViTModel.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu").eval()

#  Ekstrakcja cech
def extract_resnet_features(path):
    return resnet_model.predict(
        preprocess_input(np.expand_dims(img_to_array(load_img(path, target_size=(224, 224))), 0))
    ).flatten()

def extract_vit_features(path):
    with torch.no_grad():
        inputs = vit_processor(images=Image.open(path).convert("RGB"), return_tensors="pt").to(vit_model.device)
        return vit_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy().squeeze()

def extract_combined_features(path):
    return np.concatenate((extract_vit_features(path), extract_resnet_features(path)))

# Pomoc
def cluster_and_save(images, embeddings, method, folder, eps=5.6, min_samples=2):
    if method == "dbscan":
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "affinity":
        clustering = AffinityPropagation(random_state=0)
    else:
        raise ValueError(f"Unsupported method: {method}")
    labels = clustering.fit_predict(embeddings)
    for i, img_path in enumerate(images):
        lbl = labels[i] if labels[i] != -1 else "outliers"
        subfolder = os.path.join(folder, f"cluster_{lbl}")
        os.makedirs(subfolder, exist_ok=True)
        shutil.move(img_path, os.path.join(subfolder, os.path.basename(img_path)))

def cluster_and_save_som(images, embeddings, folder, som_size=(2,4)):
    som = MiniSom(som_size[0], som_size[1], embeddings.shape[1], sigma=1.0, learning_rate=0.0047, random_seed=42)
    embeddings = StandardScaler().fit_transform(embeddings)
    som.train_random(embeddings, 500)
    labels = [som.winner(e) for e in embeddings]
    label_map = {lbl:i for i,lbl in enumerate(set(labels))}
    for i, img_path in enumerate(images):
        subfolder = os.path.join(folder, f"cluster_{label_map[labels[i]]}")
        os.makedirs(subfolder, exist_ok=True)
        shutil.move(img_path, os.path.join(subfolder, os.path.basename(img_path)))

def process_folder(folder, method="dbscan", eps=5.6, min_samples=2, som_x=4, som_y=3, som_iterations=6625, n_neighbors=2):
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not images:
        print(f"No images found in folder: {folder}")
        return

    print(f"Processing folder: {folder} using method: {method}")
    embeddings = np.vstack([extract_vit_features(img) for img in images])

    if method == "som":
        refine_clusters_using_knn_with_som(folder, som_x=som_x, som_y=som_y, som_iterations=som_iterations, n_neighbors=n_neighbors)
    elif method in ["dbscan", "affinity"]:
        cluster_and_save(images, embeddings, method, folder, eps, min_samples)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

def move_folder_contents(src, dst):
    if os.path.exists(src):
        for item in os.listdir(src):
            shutil.move(os.path.join(src, item), os.path.join(dst, item))
        shutil.rmtree(src, ignore_errors=True)

def unpack_nested_clusters(folder):
    for root, dirs, _ in os.walk(folder):
        for d in dirs:
            cpath = os.path.join(root, d)
            if d.startswith("cluster_"):
                for item in os.listdir(cpath):
                    ipath = os.path.join(cpath, item)
                    if os.path.isdir(ipath) and os.path.basename(ipath).startswith("cluster_"):
                        for nest in os.listdir(ipath):
                            shutil.move(os.path.join(ipath, nest), cpath)
                        os.rmdir(ipath)

def refine_clusters_using_knn_with_som(folder, som_x=4, som_y=3, som_iterations=6625, n_neighbors=2):
    print("Refining clusters using SOM and KNN...")

    image_files = []
    for root, _, files in os.walk(folder):
        image_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(('jpg', 'jpeg', 'png'))])

    if not image_files:
        print(f"No valid images found in {folder}. Skipping refinement.")
        return

    embeddings = []
    for img in image_files:
        try:
            embeddings.append(extract_combined_features(img))
        except Exception as e:
            print(f"Error extracting features for {img}: {e}")

    embeddings = np.array(embeddings)
    if embeddings.size == 0:
        print(f"No valid embeddings generated from {folder}. Skipping refinement.")
        return

    print(f"Generated embeddings shape: {embeddings.shape}")

    embeddings_normalized = StandardScaler().fit_transform(embeddings)

    # Trenuj SOM
    som = MiniSom(x=som_x, y=som_y, input_len=embeddings_normalized.shape[1],
                  sigma=0.001, learning_rate=0.05, random_seed=42)
    som.random_weights_init(embeddings_normalized)
    som.train_random(embeddings_normalized, som_iterations)

    # Map features to SOM grid coordinates
    som_coords = np.array([som.winner(feature) for feature in embeddings_normalized])

    # KNN refinement
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(som_coords)
    distances, indices = knn.kneighbors(som_coords)

    refined_clusters = {}
    for img, coord in zip(image_files, indices):
        cluster_id = tuple(coord)
        refined_clusters.setdefault(cluster_id, []).append(img)

    print("Saving refined clusters to folders...")
    for cluster_id, imgs in refined_clusters.items():
        cluster_folder = os.path.join(folder, f"refined_cluster_{'_'.join(map(str, cluster_id))}")
        os.makedirs(cluster_folder, exist_ok=True)
        for img in imgs:
            if os.path.exists(img):
                shutil.move(img, os.path.join(cluster_folder, os.path.basename(img)))
            else:
                print(f"Warning: File {img} not found. Skipping.")

    print("Final refined clusters:")
    for cluster_id, imgs in refined_clusters.items():
        print(f"Cluster {cluster_id}:")
        for img in imgs:
            print(f"  - {os.path.basename(img)}")

def unpack_all_folders(root_folder):
    for root, dirs, files in os.walk(root_folder, topdown=False):
        for f in files:
            fp = os.path.join(root, f)
            dst = os.path.join(root_folder, f)
            if not os.path.exists(dst):
                shutil.move(fp, dst)
        for d in dirs:
            dp = os.path.join(root, d)
            if not os.listdir(dp):
                os.rmdir(dp)

def calculate_folder_similarity(folder):

    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    if not images:
        return np.array([])

    embeddings = []
    for img_path in images:
        try:
            embeddings.append(extract_combined_features(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if len(embeddings) < 2:
        return np.array([])

    embeddings = np.array(embeddings)

    # Obliczanie cos similarity
    similarity_matrix = np.dot(embeddings, embeddings.T)
    norms = np.linalg.norm(embeddings, axis=1)
    similarity_matrix /= np.outer(norms, norms)

    return similarity_matrix.flatten()

def find_most_and_least_similar_folders(root_folder):

    least_similar_folder = None
    most_similar_folder = None
    least_similarity_score = float('inf')
    most_similarity_score = -float('inf')

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"Calculating similarity for folder: {folder_path}")

        similarity_scores = calculate_folder_similarity(folder_path)

        if similarity_scores is None or len(similarity_scores) == 0:
            print(f"Skipping {folder_path} due to no valid similarity scores.")
            continue

        avg_similarity = np.mean(similarity_scores)

        # Najmniejsze podobienstwo
        if avg_similarity < least_similarity_score:
            least_similarity_score = avg_similarity
            least_similar_folder = folder_path

        # Najwieksze podobienstwo
        if avg_similarity > most_similarity_score:
            most_similarity_score = avg_similarity
            most_similar_folder = folder_path

    if least_similar_folder is not None and most_similar_folder is not None:
        return least_similar_folder, least_similarity_score, most_similar_folder, most_similarity_score
    else:
        print("No valid folders found to compare similarity.")
        return None

if os.path.exists(RESULTS_PATH):
    shutil.rmtree(RESULTS_PATH)
os.makedirs(RESULTS_PATH)

# Krok 1: Wstępne klastrowanie za pomocą ResNet + DBSCAN
all_imgs = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH)
            if f.lower().endswith(('jpg','jpeg','png'))]
if not all_imgs:
    print("No images found."); exit(1)

feat = np.vstack([extract_resnet_features(i) for i in all_imgs])
labels = DBSCAN(eps=44.2, min_samples=2).fit_predict(feat)

rest_folder = os.path.join(RESULTS_PATH, "rest")
cluster_outliers_folder = os.path.join(RESULTS_PATH, "cluster_outliers")
outliers_folder = os.path.join(RESULTS_PATH, "outliers")
os.makedirs(rest_folder, exist_ok=True)
os.makedirs(cluster_outliers_folder, exist_ok=True)
os.makedirs(outliers_folder, exist_ok=True)
root_folder = cluster_outliers_folder
for img, lbl in zip(all_imgs, labels):
    dest = cluster_outliers_folder if lbl==-1 else rest_folder
    shutil.copy(img, os.path.join(dest, os.path.basename(img)))

# Krok 2: Outliers przy użyciu ViT + Affinity
process_folder(cluster_outliers_folder, method="affinity")
least_similar_folder, least_similarity_score, most_similar_folder, most_similarity_score = find_most_and_least_similar_folders(root_folder)

print(f"Folder with the least similarity: {least_similar_folder} with similarity score: {least_similarity_score}")
print(f"Folder with the most similarity: {most_similar_folder} with similarity score: {most_similarity_score}")

if most_similar_folder:
    c1 = most_similar_folder
    print(f"The most similar folder is: {c1} with similarity score: {most_similarity_score}")
    move_folder_contents(c1, rest_folder)
else:
    print("No folder with valid similarity found.")

# Krok 4: Reszta przez "affinity"
process_folder(rest_folder, method="affinity")

# Krok 5: Dalsze klastrowanie
def process_and_merge_singletons(x, cluster_folder, target_folder):
    for root, dirs, _ in os.walk(cluster_folder):
        for subdir in dirs:
            dir_path = os.path.join(root, subdir)
            images = sorted(os.listdir(dir_path))
            image_count = len(images)

            if x == 1 and image_count == 1:  # Tylko pojedyncze
                target_subfolder = os.path.join(target_folder, "cluster_singleton")
                os.makedirs(target_subfolder, exist_ok=True)
                for image in images:
                    shutil.move(os.path.join(dir_path, image), os.path.join(target_subfolder, image))
                os.rmdir(dir_path)
                print(f"Moved {image_count} image(s) from {dir_path} to {target_subfolder}")

            elif x == 0 and image_count > 1:  # Wiecej niz jedn w folderze
                target_subfolder = os.path.join(target_folder, f"cluster_{subdir}")
                os.makedirs(target_subfolder, exist_ok=True)
                for image in images:
                    shutil.move(os.path.join(dir_path, image), os.path.join(target_subfolder, image))
                os.rmdir(dir_path)
                print(f"Moved {image_count} image(s) from {dir_path} to {target_subfolder}")
root_folder = rest_folder

least_similar_folder, least_similarity_score, most_similar_folder, most_similarity_score = find_most_and_least_similar_folders(root_folder)

print(f"Folder with the least similarity: {least_similar_folder} with similarity score: {least_similarity_score}")
print(f"Folder with the most similarity: {most_similar_folder} with similarity score: {most_similarity_score}")
#Klastrowanie folderu z najmniejszym similar
if least_similar_folder:
    c7 = least_similar_folder
    print(f"The least similar folder is: {c7} with similarity score: {least_similar_folder}")
    process_folder(c7, method="dbscan", eps=5.6, min_samples=1)
    unpack_all_folders(cluster_outliers_folder)
else:
    print("No folder with valid similarity found.")

process_and_merge_singletons(x=1, cluster_folder=c7, target_folder=cluster_outliers_folder)
unpack_all_folders(cluster_outliers_folder)
process_folder(cluster_outliers_folder, eps=5.65, min_samples=2)
out = os.path.join(cluster_outliers_folder, "cluster_outliers")

if os.path.exists(out):
    process_folder(out, method="som", som_x=4, som_y=3, som_iterations=6625, n_neighbors=2)
    process_and_merge_singletons(x=0, cluster_folder=out, target_folder=cluster_outliers_folder)
    move_folder_contents(os.path.join(out), outliers_folder)

unpack_all_folders(cluster_outliers_folder)
process_folder(cluster_outliers_folder, method="som", som_x=1, som_y=2, som_iterations=1000, n_neighbors=2)
process_and_merge_singletons(x=1, cluster_folder=cluster_outliers_folder, target_folder=outliers_folder)
unpack_all_folders(cluster_outliers_folder)
process_folder(cluster_outliers_folder, method="som", som_x=3, som_y=1, som_iterations=1000, n_neighbors=3)
process_and_merge_singletons(x=1, cluster_folder=cluster_outliers_folder, target_folder=outliers_folder)
unpack_all_folders(outliers_folder)
move_folder_contents(cluster_outliers_folder, rest_folder)
unpack_all_folders(rest_folder)

images = [os.path.join(rest_folder, f) for f in os.listdir(rest_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

if images:
    embeddings = np.vstack([extract_combined_features(img) for img in images])

    # Kategoryzacja fianlna reszty
    som_size = (10, 10)
    cluster_and_save_som(images, embeddings, rest_folder, som_size=som_size)

    print(f"Clustering complete. Results saved in subfolders under {rest_folder}")
else:
    print(f"No images found in {rest_folder}")

print("=== FINAL STEP COMPLETE ===")
print(f"Final processed data in: {RESULTS_PATH}")
