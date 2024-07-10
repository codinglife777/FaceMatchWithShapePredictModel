import cv2
import dlib
import numpy as np
import argparse
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None, img
    landmarks = predictor(gray, faces[0])
    shape = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        shape[i] = (landmarks.part(i).x, landmarks.part(i).y)
    for (x, y) in shape:
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
    return shape, img

def calculate_matching_percentage(landmarks1, landmarks2):
    distances = np.linalg.norm(landmarks1 - landmarks2, axis=1)
    max_distance = np.sqrt((landmarks1.max(axis=0) - landmarks1.min(axis=0))**2).sum()
    mean_distance = np.mean(distances)
    matching_percentage = (1 - mean_distance / max_distance) * 100
    return matching_percentage

def main(image_path1, image_path2):
    landmarks1, img1 = get_landmarks(image_path1)
    landmarks2, img2 = get_landmarks(image_path2)
    if landmarks1 is not None and landmarks2 is not None:
        matching_percentage = calculate_matching_percentage(landmarks1, landmarks2)
        print(f"Matching Percentage: {matching_percentage:.2f}%")
        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img1_rgb)
        axes[0].set_title('Image 1 with Landmarks')
        axes[0].axis('off')
        axes[1].imshow(img2_rgb)
        axes[1].set_title('Image 2 with Landmarks')
        axes[1].axis('off')
        plt.suptitle(f'Matching Percentage: {matching_percentage:.2f}%')
        plt.show()
    else:
        print("Could not detect faces in one or both images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face matching using landmarks.')
    parser.add_argument('image1', type=str, help='Path to the first image.')
    parser.add_argument('image2', type=str, help='Path to the second image.')
    args = parser.parse_args()
    main(args.image1, args.image2)
