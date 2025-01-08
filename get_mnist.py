import os
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
def mnist():
    # Define a directory to save the organized MNIST dataset
    output_dir = '/home/users/asubedi/neural_processes/neural_processes/neural_diffusion_process/working_dir/dataset/mnist_train'
    os.makedirs(output_dir, exist_ok=True)

    # Define transformations (convert to tensor only)
    transform = transforms.Compose([transforms.ToTensor()])

    # Download the MNIST dataset
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    to_pil = ToPILImage()

    # Organize images by class labels
    for idx, (image, label) in enumerate(mnist_train):
        # Create a directory for the class if it doesn't exist
        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)
        pil_image = to_pil(image)

        # Save the image in the respective class directory
        image_path = os.path.join(class_dir, f'{idx}.png')
        pil_image.save(image_path)

        # Optional: Print progress every 1000 images
        if idx % 1000 == 0:
            print(f"Processed {idx} images...")

    print(f"MNIST dataset organized by class saved to: {output_dir}")

mnist()