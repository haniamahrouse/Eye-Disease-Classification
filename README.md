# Eye Disease Classification



I trained two models and compared them:
1. A **CNN built from scratch**
2. A **pretrained VGG16** (transfer learning)



- Validation set: `splits/val.csv` (the only split file in the dataset).
- Train and test: I split the remaining images myself (about 1/9 of them for test).
- In ODIR each patient has a left and a right eye photo, and they look very similar.
  I kept both eyes of the same patient in the same split, so the test score is not fake-high.
- Because I made the train/test split myself, my test set is not exactly the same as the official one.

## How to run

1. Open `eye_disease_classification.ipynb` in Google Colab (choose a **GPU** runtime).
2. Upload the dataset zip file.
3. Change `ZIP_PATH` in the settings cell to the path of the zip file.
4. Run all the cells from top to bottom.

The plots and result tables are saved in the `outputs/` folder.

## What I did

**1. Data exploration**
I counted the images in each class and split, made a class distribution plot,
and looked at sample images from every class.
The dataset is very imbalanced: Normal has 4698 images but Hypertension only 88.

**2. Black border**
Many images have black pixels around the retina. I wrote a function to crop them.
Then I trained the same CNN twice, once with the border and once without, and compared the results.

**3. Class imbalance**
If we train normally, the model just predicts the big classes (Normal, DR) and ignores the small ones.
To fix this I used:
- **Weighted loss:** small classes get a bigger weight, so mistakes on them cost more.
- **Augmentation:** random flips, rotation and brightness changes on the training images.

**4. CNN from scratch**
5 blocks of Conv, BatchNorm, ReLU and MaxPool, then a small classifier with 8 outputs (softmax).

**5. VGG16 (transfer learning)**
I loaded VGG16 trained on ImageNet, froze the first conv blocks, replaced the last layer with 8 outputs,
and trained the rest with a small learning rate.

**6. Comparison**
I compared both models using accuracy, precision, recall and F1-score for every class.

## Results

(Fill these tables from the files in `outputs/` after running the notebook.)


**Which model is better?**
(VGG16 should be better because it already learned useful features from ImageNet. Write the numbers.)


