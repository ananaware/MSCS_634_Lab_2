MSCS 634 - Lab 2: Classification Using KNN and RNN Algorithms
Purpose

This lab compares K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) classifiers on the Wine dataset from scikit-learn, using an 80/20 train/test split. I tested different values of k and different radius values to see how they change accuracy.

Key Insights
KNN did slightly better than RNN. Its best accuracy was 77.8% (k = 1 and k = 21), while RNN's best was 75.0% (radius 350).
KNN accuracy went up and down as k changed, and RNN accuracy stayed flat at 72.2% from radius 400 to 600.
The gap between the models is only one or two test samples (the test set has 36), so neither model is clearly better.
Challenges and Decisions
I did not scale the features, because the radius values (350-600) only make sense on the raw data. This likely limits accuracy for both models.
I set outlier_label='most_frequent' in RNN so the model doesn't fail if a test point has no neighbors inside the radius.
Files
MSCS_634_Lab_2.ipynb - notebook with all code, plots, and explanations
