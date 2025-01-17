import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import InceptionResnetV1
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
from PIL import Image, ImageEnhance
import warnings
from sklearn.utils.multiclass import unique_labels
warnings.filterwarnings("ignore", category=FutureWarning)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained FaceNet model
print("Loading the pre-trained FaceNet model...")
model = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Define image transformations
base_transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Ensure the correct size
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize as expected by the model
])

def augment_image(image):
    """Apply data augmentation to the given PIL image."""
    augmentations = []

    # Horizontal flip
    augmentations.append(image.transpose(Image.FLIP_LEFT_RIGHT))

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    augmentations.append(enhancer.enhance(1.5))  # Increase brightness

    # Rotation
    augmentations.append(image.rotate(15))  # Rotate 15 degrees
    augmentations.append(image.rotate(-15))  # Rotate -15 degrees

    # Cropping
    width, height = image.size
    crop_size = int(0.9 * min(width, height))
    augmentations.append(image.crop((0, 0, crop_size, crop_size)).resize((160, 160)))

    return augmentations

def extract_features(image_path):
    """Extract embedding features for a face image using FaceNet."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_pil = Image.fromarray(img_rgb)  # Convert NumPy array to PIL image

    # Apply base transform and extract features
    img_tensor = base_transform(img_pil).unsqueeze(0).to(device)  # Transform, add batch dimension, and move to device
    with torch.no_grad():
        embeddings = model(img_tensor).squeeze().cpu().numpy()

    return embeddings

def load_dataset(dataset_path):
    """Load dataset and extract embeddings and labels with augmentation."""
    embeddings = []
    labels = []

    for root, _, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for file in files:
            if not file.lower().endswith(('.jpg', '.png', '.jpeg')):  # Validate image extensions
                continue
            file_path = os.path.join(root, file)
            try:
                # Load and process the original image
                img = cv2.imread(file_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)

                # Extract features from the original image
                embedding = extract_features(file_path)
                embeddings.append(embedding)
                labels.append(label)

                # Generate augmented images and extract their features
                augmented_images = augment_image(img_pil)
                for aug_img in augmented_images:
                    aug_tensor = base_transform(aug_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        aug_embedding = model(aug_tensor).squeeze().cpu().numpy()
                    embeddings.append(aug_embedding)
                    labels.append(label)

            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    return np.array(embeddings), labels

def train_classifier(X_train, y_train):
    """Train an SVM classifier."""
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train, y_train)
    return clf

def predict_face(image_path, classifier, label_decoder):
    """Predict the face identity by comparing embeddings."""
    input_embedding = extract_features(image_path)
    prediction = classifier.predict([input_embedding])[0]
    predicted_label = label_decoder[prediction]

    similarity_score = max(classifier.predict_proba([input_embedding])[0])

    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label} (Similarity: {similarity_score:.2f})")
    plt.axis('off')
    plt.show()

    return predicted_label

if __name__ == "__main__":
    # Path to the LFW dataset
    dataset_path = r"C:\Users\yashm\OneDrive\Documents\Yash\my_folder\lfw-deepfunneled\lfw-deepfunneled"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    print("Loading dataset and extracting features...")
    embeddings, labels = load_dataset(dataset_path)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Create a mapping for decoding labels
    label_decoder = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, encoded_labels, test_size=0.25, random_state=42
    )

    print("Training classifier...")
    classifier = train_classifier(X_train, y_train)

    print("Evaluating classifier...")
    y_pred = classifier.predict(X_test)

    # Ensure target_names matches unique labels in y_test
    unique_classes = unique_labels(y_test, y_pred)  # Find classes used in this dataset split
    target_names = [label_decoder[class_idx] for class_idx in unique_classes]  # Decode to class names

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred, target_names=target_names, labels=unique_classes
    ))

    # Predict for a test image
    test_image_path = r"C:\Users\yashm\OneDrive\Documents\Yash\my_folder\Test_img.jpg"
    try:
        predict_face(test_image_path, classifier, label_decoder)
    except Exception as e:
        print(f"Error during prediction: {e}")
