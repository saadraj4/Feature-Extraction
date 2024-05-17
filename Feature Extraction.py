import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
def load_images_and_labels(image_paths, labels):
    images = []
    features_list = []
    for image_path, label in zip(image_paths, labels):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append((img, label))
    return images
#%%
def extract_surf_features(images):
    surf = cv2.xfeatures2d.SURF_create()
    descriptors_list = []
    labels = []
    
    for img, label in images:
        keypoints, descriptors = surf.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            labels.append(label)
    return descriptors_list, labels
#%%
image = ""
descriptors_list, labels  = extract_surf_features(image)

def convert_descriptors_to_feature_vector(descriptors_list):
    # Simple method: just average all descriptors in an image
    return np.array([np.mean(desc, axis=0) for desc in descriptors_list])


# Prepare training data
X_train, X_test, y_train, y_test = train_test_split(descriptors_list, labels, test_size=0.3, random_state=42)

# Flatten the descriptors
X_train_flat = convert_descriptors_to_feature_vector(X_train)
X_test_flat = convert_descriptors_to_feature_vector(X_test)

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_flat, y_train)
#%%
y_pred = knn.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
