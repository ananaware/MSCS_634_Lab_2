# MSCS_634_Lab_2
MSCS 634 - Lab 2: Classification Using KNN and RNN Algorithms
Purpose

The goal of this lab was to compare two neighbor-based classifiers, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), using the Wine dataset from scikit-learn. The dataset has 178 wine samples, 13 chemical features, and 3 wine classes. I tested different values of k for KNN and different radius values for RNN to see how these choices change the accuracy, and then compared the two models.

Setup
Dataset: Wine dataset from sklearn.datasets (class counts: 59, 71, and 48)
Split: 80% training (142 samples) and 20% testing (36 samples), using random_state=42
KNN values of k tested: 1, 5, 11, 15, 21
RNN radius values tested: 350, 400, 450, 500, 550, 600
Results
k (KNN)	Accuracy		Radius (RNN)	Accuracy
1	0.7778		350	0.7500
5	0.7222		400	0.7222
11	0.7500		450	0.7222
15	0.7500		500	0.7222
21	0.7778		550	0.7222
			600	0.7222
Key Insights
KNN did slightly better than RNN overall. Its best accuracy was 77.8% (at k = 1 and k = 21), compared with 75.0% for RNN (at radius 350). The average accuracy was about 75.6% for KNN and 72.7% for RNN.
KNN accuracy went up and down as k changed. It dipped at k = 5 (72.2%) and rose again at k = 21, so there was no clear pattern where a bigger or smaller k was always better.
RNN accuracy was highest at the smallest radius (350) and then stayed flat at 72.2% from radius 400 to 600. Once the radius got large enough, adding more neighbors did not change the predictions.
The difference between the two models is small. The test set only has 36 samples, so one sample is worth about 2.8% accuracy. This means the gap is only one or two predictions, and I would not say one model is clearly better.
KNN is a good choice when data is spread out evenly and you want a simple setting to tune. RNN is better when the data is packed differently in different areas, or when only points within a certain distance should count as neighbors.
Challenges and Decisions
No feature scaling: I did not scale the features. The assignment uses radius values between 350 and 600, which only make sense on the raw data (the proline feature has values in the hundreds and thousands). I used unscaled data for both models to keep the comparison fair. The downside is that large-valued features like proline have more influence on distances, which probably limits the accuracy of both models.
Empty neighborhoods in RNN: RNN can fail if a test point has no training points inside the radius. I set outlier_label='most_frequent' so the model guesses the most common class in that case. In my run it did not change any results, but it makes the code safer.
Small test set: With only 36 test samples, the accuracy numbers can change a lot from a single prediction, so I was careful not to over-interpret small differences.
Files in This Repository
MSCS_634_Lab_2.ipynb - the Jupyter Notebook with all code, plots, and explanations
