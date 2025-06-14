from torchvision import transforms
from PIL import Image, ImageFilter, ImageOps
import random
import shutil
import torch
import cv2
import os


# Function to add salt and pepper noise to an image tensor
def add_salt_and_pepper_noise(image, prob=0.05):
    output = image.clone()
    salt_pepper = torch.rand_like(image)
    output[salt_pepper < (prob / 2)] = 0  # Pepper
    output[salt_pepper > 1 - (prob / 2)] = 1  # Salt
    return output


def add_gaussian_noise(image, mean=0.0, std=0.1):
    """
    Add Gaussian noise to an image tensor.

    Parameters:
    - image: Tensor of shape (C, H, W) representing the image.
    - mean: Mean of the Gaussian noise.
    - std: Standard deviation of the Gaussian noise.

    Returns:
    - Noisy image tensor.
    """
    # Create a tensor of Gaussian noise
    noise = torch.normal(mean=mean, std=std, size=image.size()).to(image.device)

    # Add the noise to the image
    noisy_image = image + noise

    # Clip the values to be in the valid range [0, 1]
    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image


# Define the augmentation pipeline with probabilities
augmentation_pipeline = transforms.Compose([
    transforms.ColorJitter(brightness=0.15, contrast=0.25, saturation=0.25, hue=0.1),
    transforms.Lambda(lambda img: img.filter(ImageFilter.MedianFilter(size=3)) if random.random() < 0.3 else img),
    transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN) if random.random() < 0.2 else img),
    transforms.Lambda(lambda img: ImageOps.invert(img) if random.random() < 0.5 else img),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: add_gaussian_noise(img, mean=0, std=0.1) if random.random() < 0.5 else img),
    transforms.ToPILImage(),
])


def augment_image(image_path, num_replicas):
    global augmentation_pipeline
    image = Image.open(image_path)
    augmented_images = []
    for _ in range(num_replicas):
        augmented_image = augmentation_pipeline(image)
        augmented_images.append(augmented_image)
    return augmented_images


def copy_bounding_boxes(txt_path, output_dir, num_replicas):
    for i in range(num_replicas):
        shutil.copy(txt_path, os.path.join(output_dir, f'augmented_image_{i}.txt'))


def augment_images_in_folder(folder_path, num_replicas):
    images = folder_path + 'images/'
    texts = folder_path + 'labels/'
    for filename in os.listdir(images):
        if filename.endswith('.jpg'):
            base_name = os.path.splitext(os.path.basename(filename))[0]
            image_path = os.path.join(images, filename)
            txt_path = os.path.join(texts, f'{base_name}.txt')

            print(f'augmenting: {image_path}\n            {txt_path}')

            # Augment images and save them
            augmented_images = augment_image(image_path, num_replicas)
            for i, img in enumerate(augmented_images):
                img.save(os.path.join(images, f'{base_name}_augmented_{i}.jpg'))
                shutil.copy(txt_path, os.path.join(texts, f'{base_name}_augmented_{i}.txt'))


def adjust_bounding_boxes(bboxes, img_width, img_height, operation):
    new_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        if operation == 'right':
            x_center = 1 - x_center  # Mirror right
        elif operation == 'up':
            y_center = 1 - y_center  # Mirror up
        elif operation == 'left':
            x_center = 1 - x_center  # Mirror left
        new_bboxes.append([class_id, x_center, y_center, width, height])
    return new_bboxes


def save_bboxes(file_path, bboxes):
    with open(file_path, 'w') as f:
        for bbox in bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')


def mirror_image(image_path, bboxes, base_directory):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # Mirror right
    mirrored_right = cv2.flip(image, 1)
    bboxes_right = adjust_bounding_boxes(bboxes, img_width, img_height, 'right')

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(f"{base_directory}train/images/{base_name}_mirrored_right.jpg", mirrored_right)
    save_bboxes(f"{base_directory}train/labels/{base_name}_mirrored_right.txt", bboxes_right)

    print(f'mirrored {base_name}')


def load_bboxes(label_path):
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            parts[0] = int(parts[0])
            bboxes.append(parts)
    return bboxes


def process_images(images_path, labels_path, base_directory):
    # image_files = glob.glob(os.path.join(images_path, '*.jpg'))  # Adjust the extension if needed
    for image_file in os.listdir(images_path):
        if image_file.endswith('.jpg'):
            base_name = os.path.splitext(os.path.basename(image_file))[0]
            label_file = os.path.join(labels_path, f"{base_name}.txt")
            image_file = os.path.join(images_path, image_file)
            if os.path.exists(label_file):
                bboxes = load_bboxes(label_file)
                mirror_image(image_file, bboxes, base_directory)
            else:
                print(f'AARRRRRRRRRRRRRRRRRRRRRRRRRRR, {image_file} does not have label.')


def draw_bounding_boxes(image_path, label_path):
    # Load the image
    image = cv2.imread(image_path)

    # Load bounding boxes
    bboxes = load_bboxes(label_path)

    img_height, img_width = image.shape[:2]

    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox

        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # Draw rectangle on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle

    # Show the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def move_random_files(src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir, num_files):
    # Get list of all image files in the source images directory
    image_files = [f for f in os.listdir(src_images_dir) if f.endswith('.jpg')]

    # Randomly select the specified number of image files
    selected_files = random.sample(image_files, num_files)

    # Move the selected image files and their corresponding label files
    for file_name in selected_files:
        # Move image file
        src_image_path = os.path.join(src_images_dir, file_name)
        dest_image_path = os.path.join(dest_images_dir, file_name)
        shutil.move(src_image_path, dest_image_path)

        # Move corresponding label file
        label_file_name = os.path.splitext(file_name)[0] + '.txt'
        src_label_path = os.path.join(src_labels_dir, label_file_name)
        dest_label_path = os.path.join(dest_labels_dir, label_file_name)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dest_label_path)
        else:
            print(f'AAAAAAAAAAAAAAARRRRRRRRRR, {src_image_path} does not have label')

    print('split done.')


if __name__ == '__main__':
    src_images_dir = './Data set raw/train/images'
    src_labels_dir = './Data set raw/train/labels'
    dest_images_dir = './Data set raw/val/images'
    dest_labels_dir = './Data set raw/val/labels'
    move_random_files(src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir, num_files=30)

    process_images('./Data set raw/train/images', './Data set raw/train/labels', './Data set raw/')
    # image_path = '../train/images/0_jpg.rf.00aa70acccbe6c2ea0a30a637e789d23_augmented_2.jpg'  # Replace with your image path
    # label_path = '../train/labels/0_jpg.rf.00aa70acccbe6c2ea0a30a637e789d23_augmented_2.txt'  # Replace with your label path
    # draw_bounding_boxes(image_path, label_path)

    folder_path = './Data set raw/train/'
    augment_images_in_folder(folder_path, 12)


