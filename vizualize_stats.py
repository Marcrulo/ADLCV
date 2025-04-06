import torch
import matplotlib.pyplot as plt

def plot_accuracies(stats, acc=True):

    if acc:
        train = stats['train_accs']
        val = stats['val_accs']

        print(train)
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train) + 1)
        
        plt.plot(epochs, train, 'b-', label='Training Accuracy')
        plt.plot(epochs, val, 'r-', label='Validation Accuracy')
        
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    else:
        train = stats['train_losses']
        val = stats['val_losses']
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train) + 1)
        
        plt.plot(epochs, train, 'b-', label='Training Loss')
        plt.plot(epochs, val, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


stats = torch.load('stats.pth')

# Call the function with your loaded accuracies
plot_accuracies(stats, acc=True)