# 202112-31-Credit-Card-Fraud-detection-via-Cluster-based-Scoring-and-Anomaly-Detection

Team Members: Vedant Kumar (vrk2109), Siddharth Nijhawan (sn2951), Sushant Tiwari (st3425)

## Description

The repository contains 4 jupyter notebooks containing end-to-end pipelines of implementing various iterative and clustering based anomaly detection algorithms on the dataset of Credit Card Fraud Detection 

Dataset is available here: https://www.kaggle.com/mlg-ulb/creditcardfraud

1. **data_analysis.ipynb** - performs initial data analysis by generating statistical metrics for each feature dimension like mean, std, min-max values, etc. Notebook also generates histograms for each feature vector and plots correlation heatmap as well

2. **kmeans.ipynb** - runs Kmeans clustering on the given dataset to generate consistency scores using the following methodology:

- Run K-means algorithm 10 times.
- Every run takes bootstrapped samples which are normalised between 0 and 1.
- K is varied between 0 and 20 and cluster indices, cluster centroids and number of data points in the clusters are calculated. 
- Finally, a weighted score for the data point for each combination of the assigned cluster is computed by calculating dot products of the C centroids.
- Precision-Recall Curves, ROC Curves, and AUPRC, AUROC, Scatter Plots are generated

3. **isolation_forest.ipynb** - runs Isolation Forest algorithm on the given dataset to generate anomaly scores using the following methodology:

- Isolation Forest algorithm is run 10 times.
- Every run takes bootstrapped samples with no. of trees = 100
- Scikit Learn’s inbuilt isolation forest class is used to generate isolation trees on our data set.
- decision_function() and predict() functions generate scores & predicted labels respectively.
- Outlier fraction (ratio of fraudulent to non-fraudulent transactions) is passed to the isolation forest class.
- Precision-Recall Curves, ROC Curves, and AUPRC, AUROC, Scatter Plots are generated.

4. **local_outlier_factor.ipynb** - runs Local Outlier Factor algorithm on the given dataset to generate anomaly scores using the following methodology:

- Local Outlier Factor algorithm is run 10 times .
- Computes LOF(X) = (sum of avg. LRD of X’s neighbors)/ LRD(X)
- LRD(X) = Local Reachability Distance (X) = 1/(Avg. Reachability of X from neighbors)
- Scores and predictions are generated using negative_outlier_factor_ object and fit_predict() functions of LOF class.
- “Minkowski” distance is used as a distance metric with the number of neighbors = 20
- Precision-Recall curves, histogram plots of score distribution, and ROC curves are plotted.
