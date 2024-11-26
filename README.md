# Deep Learning for Computer Vision: CNN, Faster R-CNN, and ViT on MNIST

Explore deep learning techniques including **CNN**, **Faster R-CNN**, and **Vision Transformer (ViT)** for image classification using the **MNIST** dataset. This lab includes model comparisons, fine-tuning with pretrained models, and performance evaluation.

---

## **Objective**

The main goal of this lab is to explore and implement different deep learning architectures for **computer vision tasks**, particularly focused on classifying the MNIST dataset. This lab covers the following models and techniques:

- **Convolutional Neural Networks (CNN)**
- **Faster R-CNN**
- **Vision Transformer (ViT)**

---

## **Tasks**

### **Part 1: CNN Classifier**
1. **CNN Model**  
   Build a CNN model using **PyTorch** to classify the MNIST dataset. The model includes layers like Convolution, Pooling, and Fully Connected layers. Hyperparameters like kernels, padding, stride, and optimizers are defined.
   
2. **Faster R-CNN Model**  
   Implement a **Faster R-CNN** model for MNIST classification.

3. **Comparison**  
   Compare the performance of both CNN and Faster R-CNN models using various metrics:
   - Accuracy
   - F1 Score
   - Loss
   - Training Time

4. **Fine-Tuning with Pretrained Models**  
   Fine-tune pretrained models like **VGG16** and **AlexNet** on the MNIST dataset and compare the results to CNN and Faster R-CNN models.

---

### **Part 2: Vision Transformer (ViT)**
1. **ViT Model**  
   Implement a **Vision Transformer (ViT)** model for MNIST image classification from scratch, based on an online tutorial.

2. **Comparison**  
   Analyze and compare the results of the ViT model with CNN and Faster R-CNN from Part 1.
   ## Conclusion and Observation

The Vision Transformer (ViT) achieved impressive results on the MNIST dataset, with a training accuracy of **95.71%** and a test accuracy of **96.87%**. While slightly less efficient than CNNs for small datasets due to its computational overhead and lack of inductive bias, ViT excelled in capturing global patterns, aided by data augmentations and a learning rate scheduler. Its strengths, such as flexibility and scalability, make it more suited for complex datasets, although it demonstrated robust performance even on a simpler task like digit classification.


---

## **Tools Used**

- **PyTorch**: Framework for building and training deep learning models.
- **Google Colab / Kaggle**: Platforms for running experiments and code.
- **GitHub**: For version control and project sharing.

---

## **Results**

- **CNN & Faster R-CNN Performance**: Evaluated using accuracy, F1 score, and training time.
- **ViT Performance**: Compared with CNN and Faster R-CNN to analyze the effectiveness of transformer-based models in computer vision.

# Model Comparison: CNN vs Vision Transformer (ViT)

| Metric           | CNN                  | Vision Transformer (ViT) |
|-------------------|----------------------|---------------------------|
| **Architecture** | Convolutional Layers with Pooling and Fully Connected Layers | Patch Embedding with Transformer Encoder |
| **Epochs**       | 5                    | 5                         |
| **Optimizer**    | Adam                 | Adam                      |
| **Learning Rate**| 0.0001               | 0.0005                    |
| **Train Accuracy**| ~98.0%              | ~95.7%                    |
| **Test Accuracy** | 97.87%              | 96.87%                    |
| **Loss Function** | CrossEntropyLoss     | CrossEntropyLoss          |
| **Strengths**    | Efficient and fast convergence | Handles larger context and long-term dependencies |
| **Weaknesses**   | Limited in capturing global context | Slower training due to complexity |
| **Use Case**     | Best for smaller datasets like MNIST | Useful for complex datasets requiring global feature extraction |

## Observations:
- The **CNN model** achieves slightly higher accuracy on MNIST with faster training and fewer resources required.
- The **Vision Transformer** demonstrates strong performance, leveraging transformer-based architectures to capture global relationships but at the cost of longer training time.


---

## **Conclusion**

This lab demonstrates how different deep learning models (**CNN**, **Faster R-CNN**, and **Vision Transformer**) can be applied to image classification tasks. The comparisons help evaluate which model performs best for the MNIST dataset.

---

## **Getting Started**

To run the code, follow these steps:

### **Prerequisites**
- Python 3.x
- PyTorch
- Kaggle environment (for running the notebooks)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/hafsakzoun/Deep-Learning-for-CV-CNN-Faster-R-CNN-and-ViT-on-MNIST.git
