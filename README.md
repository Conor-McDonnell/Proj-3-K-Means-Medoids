# Proj-3-K-Means-Medoids

Clustering in Unsupervised Machine Learning
This repository contains code to demonstrate the use of clustering algorithms, specifically K-Means and K-Medoids, for unsupervised machine learning tasks. The code uses Python and various libraries for data manipulation, visualization, and clustering.

Dependencies
The following Python libraries are required to run the code:

pandas
numpy
seaborn
matplotlib
scikit-learn
plotly
pickle
pandas_datareader
Install these libraries using pip or conda before running the code.

Dataset
The code reads a sales data sample from a CSV file ("sales_data_sample.csv") into a pandas DataFrame for analysis and machine learning algorithms. The dataset contains various features related to sales.

Data Preprocessing
Data preprocessing steps include dropping irrelevant columns with a high percentage of missing values, converting string values to uppercase, and scaling numerical features using the MinMaxScaler.

K-Means Clustering
The code uses the elbow method to determine the optimal number of clusters for K-Means clustering. After selecting the appropriate number of clusters, the K-Means algorithm is applied to the data, and the clusters are visualized using scatterplots. The Silhouette coefficient is used to evaluate the K-Means model's performance.

K-Medoids Clustering
Similarly, the elbow method is used to find the optimal number of clusters for K-Medoids clustering. The K-Medoids algorithm is then applied, and the clusters are visualized. The Silhouette coefficient is also calculated to evaluate the K-Medoids model's performance.

Results
The results show the performance of both K-Means and K-Medoids clustering algorithms on the given sales dataset. The Silhouette score is used as a metric to compare the models, and the code visualizes the clusters for better understanding.
