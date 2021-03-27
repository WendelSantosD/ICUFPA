#Load various data sets
from sklearn.datasets import load_wine
#Load classification models
from sklearn.neighbors import KNeighborsClassifier
#Dataset management
from sklearn.model_selection import train_test_split
#Evaluation metrics
from sklearn.metrics import accuracy_score # for classification
from sklearn.metrics import mean_squared_error # for regression
from sklearn.metrics import classification_report
# For data management and visualization
import pandas as pd
# More efficient numeric calculations
import numpy as np

#Load the dataset and divide it in X = features and y = labels
X, y = load_wine(return_X_y=True)

#Divide the sample dataset in train/test sets by the specified ratio
print(X.shape,y.shape)

# This method shuffles the dataset, then divides it. The "random_state"
# parameter serve as a seed for the shuffling process, allowing
# reproducibility of the results
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Use the model (KNN)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print(classification_report(y_test, y_pred))
print(f'Accuracy score:', accuracy_score(y_test, y_pred))