import os
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configurations
image_size = 224
batch_size = 64
num_workers = 0
learning_rate = 1e-4
num_epochs = 20
unfreeze_epoch = 5
class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']

# Define transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(15),  # Increased rotation for more variability
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Larger translation
    transforms.Resize((image_size, image_size)),  # Fixed size
    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),  # Larger random crop
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Stronger color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Standard normalization for grayscale
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # Resize to model input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale images
])

def predict_lung_disease(image, model_path='C:/Users/User/Downloads/edumit-20250414T165044Z-001/edumit/llama2-PDF-Chatbot/weights/resnet101_lung_model.pth'):
    """
    Predict the lung disease class for a given PIL image.
    Args:
        image (PIL.Image): The input image.
        model_path (str): Path to the trained model weights.
    Returns:
        str: Predicted class name.
    """
    import torch
    from torchvision import transforms
    import torch.nn as nn
    import torchvision.models as models
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['COVID', 'Normal', 'Pneumonia', 'Pneumothorax', 'Tuberculosis']
    
    # Preprocessing for a single image
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).to(device)
    
    # Load model architecture and weights
    model = models.resnet101(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, 5)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        pred_class = class_names[predicted.item()]
    return pred_class

if __name__ == "__main__":
    # Dataset paths
    dir = 'C:\\Users\\patrick\\Desktop\\Lung Disease\\data'
    print("Loading full dataset...")
    full_dataset = datasets.ImageFolder(dir, transform=train_transform)

    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size])
    test_dataset.dataset.transform = test_transform

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load model and apply Dropout
    print("Loading pre-trained ResNet101 model...")
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),  # Increased dropout to 0.7
        nn.Linear(model.fc.in_features, 5)
    )
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer with weight decay and adjusted learning rate
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=learning_rate, weight_decay=1e-5)  # Increased weight decay

    # Adjust the learning rate scheduler to decay sooner
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # Lists to store training and validation losses and accuracies
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Initially, freeze all layers except the final layer
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Training loop with progressive unfreezing
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Starting epoch {epoch + 1}")

        # Training step
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for training
        average_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(average_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}] completed. Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():  # Disable gradient tracking
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)

                running_val_loss += val_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for validation
        average_val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_losses.append(average_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Validation Loss: {average_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # Step the scheduler with validation loss (minimization)
        scheduler.step(average_val_loss)

    # Test set evaluation
    model.eval()
    running_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)

            running_test_loss += test_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    average_test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test

    print("\nTest Set Evaluation:")
    print(f"Test Loss: {average_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Improved plotting
    plt.figure(figsize=(18, 5))

    # Define epochs for x-axis
    epochs = list(range(1, num_epochs + 1))

    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs, rotation=45)
    plt.grid(True)
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', marker='o')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epochs, rotation=45)
    plt.grid(True)
    plt.legend()

    # Test Accuracy Confusion Matrix 
    # Placeholders for true labels and predicted labels
    y_true = []
    y_pred = []

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Append true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')

    # Add title and show plot
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.show()

    # Save the model
    MODEL_SAVE_PATH = 'resnet101_lung_model.pth'
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
