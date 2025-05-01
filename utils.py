# utils.py — funções auxiliares para visualização de métricas

import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss_accuracy(train_losses, val_losses, train_acc, val_acc):
    os.makedirs("2D_GFI_results", exist_ok=True)
    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='s')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("2D_GFI_results/loss_curve.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(train_acc, label='Train Accuracy', marker='o')
    plt.plot(val_acc, label='Val Accuracy', marker='s')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("2D_GFI_results/accuracy_curve.png", dpi=300)
    plt.show()
