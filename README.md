
# 🧠 Neural Network From Scratch (NumPy Implementation)

This project demonstrates how to build and train a **feedforward neural network** from scratch using only **NumPy**, without any high-level deep learning libraries like TensorFlow or PyTorch. The model is trained on the **MNIST digit classification** dataset and achieves competitive accuracy.

<p align="center">
  <img src="https://raw.githubusercontent.com/Vigneshmaradiya/Neural-Network-From-Scratch/main/Predicted-Actual.png" width="500"/>
</p>

---

## 🚀 Features

- Built entirely using **NumPy**
- Fully connected (dense) neural network layers
- ReLU and Softmax activation functions
- Forward and backward propagation
- Categorical Crossentropy loss
- Training loop with manual gradient updates
- Evaluation with classification report and confusion matrix
- Visualizations of predictions and performance

---

## 📊 Results

- **Test Accuracy**: 89.44%
- **F1-Score (macro avg)**: 89%

### 🔍 Classification Report

```
               precision    recall  f1-score   support

           0       1.00      0.94      0.97        35
           1       0.87      0.75      0.81        36
           2       0.97      1.00      0.99        35
           3       0.97      0.78      0.87        37
           4       0.92      0.92      0.92        37
           5       0.82      1.00      0.90        37
           6       0.97      0.95      0.96        37
           7       0.89      0.92      0.90        36
           8       0.84      0.82      0.83        33
           9       0.74      0.86      0.80        37

    accuracy                           0.89       360
   macro avg       0.90      0.89      0.89       360
weighted avg       0.90      0.89      0.89       360
```

### 📉 Confusion Matrix

<p align="center">
  <img src="https://raw.githubusercontent.com/Vigneshmaradiya/Neural-Network-From-Scratch/main/Confusion-Matrix.png" width="500"/>
</p>

---

## 🗂️ Project Structure

```
Neural-Network-From-Scratch/
│
├── activations.py       # Activation functions (ReLU, Softmax)
├── layer.py             # Dense layer class with forward/backward
├── loss.py              # Categorical Crossentropy loss
├── main.py              # Main training and evaluation script
├── requirements.txt     # Required Python packages
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Vigneshmaradiya/Neural-Network-From-Scratch.git
cd Neural-Network-From-Scratch
```

### 2. (Optional) Create a Virtual Environment

```bash
python3 -m venv nn
source nn/bin/activate  # On Linux/macOS
nn\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Model

```bash
python main.py
```

The script will:
- Load MNIST data
- Train the neural network
- Evaluate and print accuracy, classification report
- Plot predicted vs actual values and the confusion matrix

---

## 📌 Example Output

Each image shows: `Predicted / Actual`

<p align="center">
  <img src="https://raw.githubusercontent.com/Vigneshmaradiya/Neural-Network-From-Scratch/main/Predicted-Actual.png" width="500"/>
</p>

---

## 📚 Concepts Covered

- Matrix-based forward propagation
- Backpropagation with manual gradient calculation
- Weight updates using gradient descent
- Multi-layer neural architecture
- Model evaluation and performance metrics

---

## 🤝 Contributing

Want to improve this project?
- Extend to include batch training or regularization
- Improve modularity or performance
- Add support for saving/loading models

Feel free to fork the repo and submit a pull request!

---

## 📜 License

This project is licensed under the [MIT License](https://github.com/Vigneshmaradiya/Neural-Network-From-Scratch/blob/main/LICENSE).

---

## 👨‍💻 Author

**Vignesh Maradiya**  
📧 [LinkedIn](https://www.linkedin.com/in/vigneshmaradiya/)  
🌐 [GitHub](https://github.com/Vigneshmaradiya)

---

⭐ *If you found this helpful, give it a star on GitHub!*
