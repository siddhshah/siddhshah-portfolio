---
type: ProjectLayout
title: Semantic Segmentation
colors: colors-a
date: '2024-12-20'
client: ''
description: >-
  Uses a convolutional neural network to perform background-foreground
  separation on layered elements in varied-resolution media.
media:
  type: ImageBlock
  url: /images/img.png
  altText: Project image
featuredImage:
  type: ImageBlock
  url: /images/img.png
  altText: altText of the image
  caption: Caption of the image
  elementId: ''
---
This project involves semantic segmentation, a technique in computer vision or image-based machine learning where each pixel in an image is labeled or annotated.

I obtained a dataset of hundreds of traffic camera images in the suburbs of Chicago, IL, where each image had a corresponding set of annotations.

The goal of this neural network was to separate vehicles from their backgrounds, where performance was measured through Intersection over Union (IoU) metrics (a higher IoU \~ greater accuracy in vehicle detection).

These images had varied resolutions, perspectives, and densities of vehicles in different parts of the images. Some images had a greater amount of background elements (more trees, road signs, etc.) while some were more empty. This subtle variance posited the dataset as a good fit for the neural network's training loop.



### Step 1: Dataloading

```
class BFSDataset(Dataset):
def 
```

```
    self.image_paths = []
    self.annotation_paths = []

    # Collect all image and annotation paths
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(".jpg"):  # Ensure we only process image files
                image_path = os.path.join(root, file)
                annotation_path = image_path.replace("Images", "Annotations").replace(".jpg", ".png")
                self.image_paths.append(image_path)
                self.annotation_paths.append(annotation_path)

def __len__(self):
    return len(self.image_paths)

def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    annotation_path = self.annotation_paths[idx]

    image = Image.open(image_path).convert("RGB")
    annotation = Image.open(annotation_path).convert("L")  # Grayscale mask

    if self.image_transform:
        image = self.image_transform(image)
    if self.annotation_transform:
        annotation = self.annotation_transform(annotation)

    # Convert annotation to binary
    annotation = (annotation > 0).float()  # Pixels > 0 are foreground

    return image, annotation
```



Dataset processing was relatively simple--the data was organized in a structured hierarchy that made it easy to process and convert to image and annotation data pairs. The traffic camera images were organized into four scene subsets: Buffalo Grove at Deerfield North, South, East, and West.



### Step 2: Semantic Segmentation Model

```
class SegmentationModel(nn.Module):
def 
```

```
    # Encoder for feature extraction
    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )

    # Decoder for upsampling
    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
    )

def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
```



The model, based on PyTorch/TorchVision, used a variety of feature extraction models, like batch norms, max pooling, and 2D convolution layers. This downsampled the data accordingly, after which the upsampler (decoder), with three connected layers of transposed 2D convolution layers and rectified linear units, completed the forward pass, which could then be used in the training loop for training and validation.

### Part 3: Training and Validation

```
if 
```

```
annotation_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),  # Resize annotations with nearest neighbor
    transforms.ToTensor()
])

# Directory paths
image_dir = r"C:\\Users\\siddh\\Desktop\\ECE 364\\BFSData\\Images"
annotation_dir = r"C:\\Users\\siddh\\Desktop\\ECE 364\\BFSData\\Annotations"

# Scene names
scenes = [
    "Buffalo Grove at Deerfield East",
    "Buffalo Grove at Deerfield North",
    "Buffalo Grove at Deerfield South",
    "Buffalo Grove at Deerfield West"
]

# Training and validation set loading
dataset = BFSDataset(image_dir, annotation_dir, image_transform=image_transform, annotation_transform=annotation_transform)
indices = list(range(len(dataset)))
np.random.shuffle(indices)  # Shuffle indices for randomness

half_size = len(indices) // 2
train_indices = indices[:half_size]
val_indices = indices[half_size:]

# Dataloaders for training and validation
train_loader = DataLoader(dataset, batch_size=16, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=16, sampler=SubsetRandomSampler(val_indices))

# Create per-scene datasets and dataloaders for validation IoU
per_scene_val_loaders = {}
for scene in scenes:
    scene_image_dir = os.path.join(image_dir, scene)
    scene_annotation_dir = os.path.join(annotation_dir, scene)
    scene_dataset = BFSDataset(scene_image_dir, scene_annotation_dir, image_transform=image_transform, annotation_transform=annotation_transform)
    scene_loader = DataLoader(scene_dataset, batch_size=16, shuffle=False)
    per_scene_val_loaders[scene] = scene_loader

# Model, BCE loss, Adam optimizer
model = SegmentationModel()
criterion = nn.BCEWithLogitsLoss()  # Using logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Use Adam optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce learning rate every 5 epochs

# Train model
trained_model, metrics = training_loop(model, train_loader, val_loader, optimizer, criterion, scheduler, device='cpu', epochs=30)
```



PyTorch Dataloaders/SubsetRandomSamplers were used to divide the data into training and validation sets of equal size. The model utilized the Adam optimizer (multiple optimizers were explored, including Adagrad and SGD, but Adam yielded the best results) and the BCEWithLogitsLoss loss criterion (multiple loss functions were explored in a similar fashion). After many trials, I settled on a learning rate of 0.0001, integrated with a learning rate scheduler that reduced this LR every 5 epochs (for a total of 50 epochs).

After this preprocessing, the data was sent to the training loop to segment the traffic camera images and extract the vehicles.

![](/images/Screenshot%202025-05-24%20233847.png)

### Notes

Although the image effectively bounded the vehicles in the traffic camera images, the resulting IoU value at the final epoch was very low, and thus worsened the prediction results. Changes in optimization, loss functions, and overall training loop structure is being explored to find fixes.



### For code and dataset:

<https://github.com/siddhshah/CNN-BFS/tree/main>

