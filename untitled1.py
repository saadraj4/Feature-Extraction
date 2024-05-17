import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_and_extract_features(image_paths, labels):
    """
    Load images from provided paths and extract SURF features.
    Args:
        image_paths (list of str): Paths to the images.
        labels (list): Corresponding labels of the images.

    Returns:
        tuple: A tuple containing lists of descriptors and corresponding labels.
    """
    # Initialize SURF detector
    surf = cv2.xfeatures2d.SURF_create(400)  # Threshold for Hessian keypoint detector
    descriptors_list = []
    valid_labels = []  # Only labels of images that have descriptors

    for image_path, label in zip(image_paths, labels):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            keypoints, descriptors = surf.detectAndCompute(img, None)
            if descriptors is not None:
                descriptors_list.append(descriptors)
                valid_labels.append(label)

    return descriptors_list, valid_labels

def convert_descriptors_to_feature_vector(descriptors_list):
    """
    Convert a list of SURF descriptors for each image to a fixed size feature vector.
    Args:
        descriptors_list (list of ndarray): List of descriptors for each image.

    Returns:
        ndarray: Array of feature vectors.
    """
    # Simple method: average all descriptors in an image
    return np.array([np.mean(desc, axis=0) for desc in descriptors_list if desc is not None])

def main():
    # Example image paths and labels
    image_paths = ['./obj1_5.JPG', '/.obj1_t1.JPG']  # Replace with actual paths
    labels = [0, 1]  # Example labels for the images

    # Load and extract features
    descriptors_list, valid_labels = load_and_extract_features(image_paths, labels)

    # Convert descriptors to feature vectors
    X = convert_descriptors_to_feature_vector(descriptors_list)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, valid_labels, test_size=0.3, random_state=42)

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict and evaluate the classifier
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
