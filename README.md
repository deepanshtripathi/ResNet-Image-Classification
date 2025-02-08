# Image Classification - Cats and Dogs

This project implements a binary image classification model using ResNet18 in PyTorch to distinguish between images of cats and dogs. The model is trained from scratch (no pre-trained weights) and evaluated using a structured pipeline.

## Project Overview
- **Dataset:** Kaggle Animal Dataset (Filtered: Only Cats & Dogs).
- **Preprocessing:** Images resized to 64x64 and converted to grayscale.
- **Model:** Custom ResNet18 with modified input/output layers.
- **Training:** Implemented with CrossEntropyLoss & Adam Optimizer.
- **Evaluation:** Logs loss & accuracy using TensorBoard.
- **Testing:** Saves test metrics and prediction results persistently.
- **Visualization:** Training metrics are visualized using TensorBoard.

## How to Set Up & Run the Project

### File Structure
```
image_classification/
│── data/
│   ├── raw/ (Original dataset)
│   ├── processed/ (Preprocessed images)
│── checkpoint/
│   ├── best_model.pth (Saved trained model)
│── logs/
│   ├── tensorboard/ (Training logs)
│   ├── training_metrics.csv (Epoch-wise loss & accuracy)
│   ├── test_metrics.csv (Final test results)
│   ├── predictions.csv (Saved single-image predictions)
│── models/
│   ├── train.py (Model training script)
│   ├── test.py (Model evaluation script)
│   ├── predict.py (Single image inference)
│   ├── resnet.py (Custom ResNet18 model)
│── utils/
│   ├── dataset.py (Preprocessing script)
│── visualizations/
│   ├── accuracy.png (Train & validation accuracy graphs)
│   ├── loss.png (Train & validation loss graphs)
README.md (Project documentation)
```

### 1. Clone or Download the Repository
```bash
git clone https://github.com/deepanshtripathi/ResNet-Image-Classification.git
cd image_classification
```
### **Important Notes**
- **The trained model (`best_model.pth`) and the raw dataset (`data/raw/`) are NOT included in this repository.**
- **You will need to:**
  1. **Download the dataset manually** and place it inside `data/raw/`.
  2. **Train the model** using `train.py` to generate `checkpoint/best_model.pth`.

### 2. Install Dependencies
Ensure Python 3.10 is installed. Then install required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Preprocess the Dataset
Ensure your raw dataset is inside `data/raw/`, then run:
```bash
python image_classification/utils/dataset.py
```
This will:
- Convert images to grayscale.
- Resize them to 64x64.
- Save processed images in `data/processed/`.

### 4. Train the Model
Run the training script:
```bash
python image_classification/models/train.py
```
This will:
- Train the model on the dataset.
- Save logs in `logs/tensorboard/` for TensorBoard visualization.
- Save the best model in `checkpoint/best_model.pth`.

### 5. View Training Results in TensorBoard
Start TensorBoard to monitor loss & accuracy:
```bash
tensorboard --logdir=logs/tensorboard
```
Open [http://localhost:6006/](http://localhost:6006/) in your browser.

### 6. Test the Model
Evaluate model performance on the test dataset:
```bash
python image_classification/models/test.py
```
Results will be saved in `logs/test_metrics.csv`.

### 7. Run Predictions on New Images
Use the trained model to classify a single image:
```bash
python image_classification/models/predict.py
```
Enter the path to an image, and it will return:
- **Predicted Class:** "Cat" or "Dog"
- **Confidence Score**
- Saves the prediction in `logs/predictions.csv`.

---

## Dataset & Preprocessing
The dataset consists of images of cats and dogs. Images undergo:
- **Grayscale conversion** (to reduce complexity).
- **Resizing to 64x64 pixels** (to ensure consistent input shape).
- **Normalization** using `mean=0.5`, `std=0.5`.

---

## Model Architecture
- **Base Model:** ResNet18
- **Input Layer:** Modified to accept 1-channel grayscale images instead of 3-channel RGB.
- **Output Layer:** Adjusted for binary classification (**Cat = 0, Dog = 1**).


### Model Summary
| Layer            | Details                        |
|-----------------|--------------------------------|
| Conv1          | 64 filters, 3x3 kernel, Stride 1 |
| ResNet Block 1 | 2x Basic Blocks |
| ResNet Block 2 | 2x Basic Blocks (Downsampling) |
| ResNet Block 3 | 2x Basic Blocks (Downsampling) |
| ResNet Block 4 | 2x Basic Blocks (Downsampling) |
| Fully Connected | 512 → 2 (Binary classification) |

---

## Training & Evaluation Results
Results after 20 epochs:
- **Train Accuracy:** ~98%
- **Validation Accuracy:** ~98%
- **Test Accuracy:** ~99.68%
- **Final Model:** Saved as `checkpoint/best_model.pth`
- **Loss & Accuracy Graphs:** View in TensorBoard: `logs/tensorboard/`

---

## Challenges Encountered & How They Were Resolved
### Choosing the Learning Rate
- **Issue:** Finding an optimal learning rate for training stability.
- **Solution:** `0.001` was chosen based on common best practices for Adam optimizer, which yielded good results.

### Ensuring Data Preprocessing Consistency
- **Issue:** Preventing mismatches between training and testing images.
- **Solution:** Standardized grayscale conversion & normalization.

### Setting Up TensorBoard for Monitoring
- **Issue:** Ensuring proper logging for loss and accuracy.
- **Solution:** Integrated TensorBoard and saved all metrics.

### Persistent Saving of Predictions & Metrics
- **Issue:** Ensuring results were not lost after running scripts.
- **Solution:** Created structured logging for:
  - `logs/training_metrics.csv`
  - `logs/test_metrics.csv`
  - `logs/predictions.csv`

---

## Final Notes
- All training & test logs are stored persistently.
- TensorBoard provides an interactive way to analyze results.
- The model achieves high accuracy without pre-trained weights.
- **Potential future improvements:** Different architectures, Hyperparameter tuning.

## 🔗 References
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

