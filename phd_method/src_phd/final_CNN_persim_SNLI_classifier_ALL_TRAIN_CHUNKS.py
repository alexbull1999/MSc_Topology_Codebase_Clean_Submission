
#!/usr/bin/env python3
"""
Simple SNLI Persistence Image CNN

This script:
1. Loads SNLI training chunks 1-5 for training
2. Uses SNLI validation set for validation  
3. Trains a CNN on 30x30 persistence images
4. Performs 3-way softmax predictions on test data

Note: You'll need to specify a test dataset path since SNLI test 
persistence images don't appear to be precomputed in your project.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
import argparse
from pathlib import Path

class PersistenceImageCNN(nn.Module):
    """CNN for persistence images (30x30)"""
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # CNN layers for 30x30 images
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 30x30 -> 30x30
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 30x30 -> 30x30
        self.pool = nn.MaxPool2d(2, 2)  # Halves dimensions
        
        # After conv1 + pool: 30 -> 15
        # After conv2 + pool: 15 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 15, 15)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class PersistenceDataset(Dataset):
    """Dataset for persistence images"""
    
    def __init__(self, persistence_images, labels):
        self.persistence_images = persistence_images
        self.labels = labels
        
    def __len__(self):
        return len(self.persistence_images)
    
    def __getitem__(self, idx):
        # Reshape to 30x30 for CNN
        p_image = self.persistence_images[idx].reshape(30, 30)
        p_image = torch.FloatTensor(p_image).unsqueeze(0)  # Add channel dimension
        
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return p_image, label

def load_training_data():
    """Load ALL 5 chunks of SNLI training persistence images"""
    
    print("Loading SNLI training chunks...")
    
    all_train_images = []
    all_train_labels = []
    
    # Load all 5 SNLI chunks
    for chunk_idx in range(1, 6):
        chunk_path = f"/vol/bitbucket/ahb24/tda_entailment_new/chunked_snli_train_persistence_images_chunk_{chunk_idx}_of_5.pkl"
        
        if Path(chunk_path).exists():
            with open(chunk_path, 'rb') as f:
                chunk_data = pickle.load(f)
            
            chunk_images = chunk_data['persistence_images']
            chunk_labels = chunk_data['labels']
            
            print(f"  Chunk {chunk_idx}: {len(chunk_images)} samples")
            
            all_train_images.append(chunk_images)
            all_train_labels.extend(chunk_labels)
        else:
            print(f"  Chunk {chunk_idx} not found: {chunk_path}")
    
    # Combine all training chunks
    if all_train_images:
        train_persistence_images = np.vstack(all_train_images)
    else:
        raise ValueError("No training chunks found!")
    
    # Convert string labels to indices if needed
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(all_train_labels[0], str):
        train_labels = np.array([label_to_idx[label] for label in all_train_labels])
    else:
        train_labels = np.array(all_train_labels)
    
    print(f"Total training samples: {len(train_persistence_images)}")
    
    return train_persistence_images, train_labels

def load_validation_data():
    """Load SNLI validation persistence images"""
    
    print("Loading SNLI validation data...")
    
    val_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_val_persistence_images.pkl"
    
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Validation file not found: {val_path}")
    
    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)
    
    val_images = val_data['persistence_images']
    val_labels = val_data['labels']
    
    # Convert string labels to indices if needed
    label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
    if isinstance(val_labels[0], str):
        val_labels = np.array([label_to_idx[label] for label in val_labels])
    else:
        val_labels = np.array(val_labels)
    
    print(f"Validation samples: {len(val_images)}")
    
    return val_images, val_labels

def load_test_data():
    """Load test persistence images from specified path"""
    
    test_path = "/vol/bitbucket/ahb24/tda_entailment_new/precomputed_snli_test_persistence_images.pkl"

    print(f"Loading test data from: {test_path}")
    
    if not Path(test_path).exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    test_images = test_data['persistence_images']
    
    # Handle labels if they exist (for evaluation)
    test_labels = None
    if 'labels' in test_data:
        test_labels = test_data['labels']
        
        # Convert string labels to indices if needed
        label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        if isinstance(test_labels[0], str):
            test_labels = np.array([label_to_idx[label] for label in test_labels])
        else:
            test_labels = np.array(test_labels)
    
    print(f"Test samples: {len(test_images)}")
    
    return test_images, test_labels

def train_model(model, train_loader, val_loader, device='cuda', epochs=50):
    """Train the CNN model"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("Training CNN...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Train Acc {train_acc:.3f}, Val Acc {val_acc:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Training completed. Best validation accuracy: {best_val_acc:.3f}")
    
    return model, best_val_acc

def predict_test_set(model, test_images, device='cuda', batch_size=64):
    """Make predictions on test set"""
    
    model.eval()
    
    # Create dataset without labels for prediction
    class TestDataset(Dataset):
        def __init__(self, images):
            self.images = images
            
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            p_image = self.images[idx].reshape(30, 30)
            p_image = torch.FloatTensor(p_image).unsqueeze(0)
            return p_image
    
    test_dataset = TestDataset(test_images)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    all_probabilities = []
    
    print("Making predictions on test set...")
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Get softmax probabilities
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)

def evaluate_if_labels_available(predictions, test_labels):
    """Evaluate predictions if test labels are available"""
    
    if test_labels is None:
        print("No test labels available for evaluation")
        return
    
    accuracy = accuracy_score(test_labels, predictions)
    
    print(f"\nTest Set Evaluation:")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Classification report
    label_names = ['entailment', 'neutral', 'contradiction']
    report = classification_report(test_labels, predictions, 
                                 target_names=label_names, 
                                 digits=3)
    print("\nClassification Report:")
    print(report)

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description="Simple SNLI Persistence CNN")
    parser.add_argument('--device', default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    args = parser.parse_args()
    
    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Load data
    train_images, train_labels = load_training_data()
    val_images, val_labels = load_validation_data()
    test_images, test_labels = load_test_data()
    
    # Create datasets and data loaders
    train_dataset = PersistenceDataset(train_images, train_labels)
    val_dataset = PersistenceDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size*2, shuffle=False)
    
    # Create and train model
    model = PersistenceImageCNN()
    trained_model, best_val_acc = train_model(model, train_loader, val_loader, 
                                            device=device, epochs=args.epochs)
    
    # Make predictions on test set
    predictions, probabilities = predict_test_set(trained_model, test_images, device=device)
    
    # Convert predictions to labels for output
    idx_to_label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    predicted_labels = [idx_to_label[pred] for pred in predictions]
    
    print(f"\nPredictions completed!")
    print(f"Test set size: {len(predictions)}")
    print(f"Prediction distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  {idx_to_label[idx]}: {count} ({count/len(predictions)*100:.1f}%)")
    
    # Evaluate if labels are available
    evaluate_if_labels_available(predictions, test_labels)
    
    # Save predictions
    results = {
        'predictions': predictions,
        'predicted_labels': predicted_labels,
        'probabilities': probabilities,
        'best_validation_accuracy': best_val_acc
    }
    
    output_path = 'snli_cnn_predictions.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()