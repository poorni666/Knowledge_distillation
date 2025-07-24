# Knowledge Distillation for Deep Learning Models

This repository contains the implementation of Knowledge Distillation to train a smaller student model (ResNet18) using a larger pre-trained and fine-tuned teacher model (ResNet50) on the CIFAR10 dataset. This project was completed as the final assignment for the Deep Learning course.

## Overview

Knowledge Distillation transfers knowledge from a large teacher model to a smaller student model, enabling efficient learning while maintaining accuracy, reducing model size and latency for resource-constrained deployment.

## Key Contributions

- **Teacher Model Fine-tuning:** Adapted the teacher model to the target dataset to provide reliable guidance for the student.
- **Robust Student Training:** Combined hard (cross-entropy), soft (KL divergence), and intermediate feature losses with hyperparameter tuning to optimize robust student model.
- **Hyperparameter Tuning:** Tuned temperature and alpha as key hyperparameters. Applied dynamic scheduling — temperature decays from 5.0 to 1.0 to gradually sharpen the teacher’s outputs, while alpha decreases to shift focus from soft to hard targets. This allows the student to learn more effectively from the teacher in early stages, then rely more on ground-truth supervision, leading to better convergence and generalization compared to fixed values.
- **Comprehensive Evaluation:** Compared teacher, student via knowledge distillation ( hyperparameter tuning) and student without distillation by measuring accuracy, parameter count, and inference latency to highlight efficiency and performance improvements.

## Distillation Methodology Used

- **Representation Alignment:** Aligning intermediate features between teacher and student to enhance knowledge transfer (feature-based KD).
- **Soft Targets:** Using the teacher’s softened output probabilities to guide the student’s learning beyond hard labels.

## Code Structure
- `main.py` —  Main script to initialize models, set hyperparameters, execute training and evaluation.
- `teacher_core.py` — Teacher training logic  
- `student_core.py` — Student training logic  
- `distillation_loss.py` — Implementation of combined loss function  
- `model_helpers.py` — Model setup and feature extraction  
- `utils.py` — Utility functions for evaluation and metrics

## Results
Models accuracy, size, latency are compared.
