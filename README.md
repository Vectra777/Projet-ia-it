# Cats vs Dogs: Image Classification with CNNs

ST2AIM project, EFREI. Students: Valentin Belougne, Simon JENNEQUIN, Thomas DANJOU.

We build a convolutional neural network (CNN) from scratch to tell cats from dogs, try two improvements (data augmentation; dropout + batch normalization), and compare them with MobileNetV2 transfer learning, frozen and fine-tuned. We also look inside the network with feature maps and Grad-CAM.

## Results (400 unseen test images)

| Model | Accuracy | F1 (dog) | ROC AUC |
|---|---|---|---|
| Baseline CNN (4 blocks, 392,481 parameters) | 77.8% | 0.757 | 0.852 |
| + Data augmentation | 74.0% | 0.746 | 0.844 |
| + Dropout + batch normalization | 74.5% | 0.736 | 0.817 |
| MobileNetV2, frozen (feature extraction) | 94.5% | 0.945 | 0.987 |
| MobileNetV2, fine-tuned (last 30 layers) | **94.8%** | **0.947** | **0.989** |

With 400 test images one standard error is about 2 points, so the three from-scratch CNNs are not clearly different. Pretraining on ImageNet is what makes the big difference; Grad-CAM shows our small CNN partly relies on image backgrounds.

## Files

- `cats_dogs_cnn.ipynb`: all code, explanations, graphs and results, with references and our AI-use statement at the end.
- `cats_dogs_cnn_report.html`: the executed notebook as a readable web page.
- `presentation/cats_dogs_cnn_presentation.pptx` and `.pdf`: the 10-minute presentation, built from the notebook's own figures.
- `presentation/speaker_notes.md`: what to say on each slide (also in the PowerPoint notes).
- `presentation/build_presentation.py`: rebuilds the figures and the deck from the executed notebook.
- `requirements.txt`: Python dependencies.

## What the notebook covers

1. **CNN basics**: convolution, ReLU, pooling, dense layers, and how training works.
2. **Data**: class counts per split, original image sizes (1,604 distinct sizes, from 59 × 41 to 1050 × 768 px), resizing to 96 × 96 × 3, sample images, stratified 80/20 train/test split.
3. **Model**: four convolutional blocks (16, 32, 64, 128 filters), parameter count computed by hand and checked against Keras, Adam + binary cross-entropy.
4. **Training**: batch size 100, 15 epochs, accuracy/loss curves, overfitting analysis; we keep the epoch with the lowest validation loss.
5. **Evaluation**: confusion matrices, accuracy, precision, recall, F1, ROC curves.
6. **Improvements**: data augmentation (train and validation only, never test), dropout + batch normalization, each applied separately.
7. **Bonus**: MobileNetV2 transfer learning and fine-tuning, error analysis, feature maps, Grad-CAM.

## Data split

We use 1,600 training images, 1,000 validation images and 400 test images, with equal numbers of cats and dogs everywhere. We train on training data, select weights and models on validation data, and use the test set only once at the end.

## Run

```bash
python -m pip install -r requirements.txt
```

Open `cats_dogs_cnn.ipynb` in Jupyter or Colab and run all cells from the project root (restart the kernel before rerunning). The dataset and the pretrained MobileNetV2 weights download automatically on the first run. On a CPU the whole notebook takes about 2–3 minutes.

To rebuild the presentation after rerunning the notebook:

```bash
python presentation/build_presentation.py
soffice --headless --convert-to pdf --outdir presentation presentation/cats_dogs_cnn_presentation.pptx
```

Multi-threaded CPU training is not perfectly deterministic, so a rerun can move the dropout + batch-normalization numbers by about one point.

## Sources and use of AI

All sources (dataset, TensorFlow/Keras documentation, MobileNetV2, Adam, batch normalization, dropout and Grad-CAM papers) are listed in the last cell of the notebook. We used an AI assistant only for hyperparameter recommendations, for questioning our choices and interpretations, and for text formatting.
