from sklearn.datasets import load_wine
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names'])
# (178, 13)
# (178,)
# .. _wine_dataset:

# Wine recognition dataset
# ------------------------

# **Data Set Characteristics:**

# :Number of Instances: 178
# :Number of Attributes: 13 numeric, predictive attributes and the class
# :Attribute Information:
#     - Alcohol
#     - Malic acid
#     - Ash
#     - Alcalinity of ash
#     - Magnesium
#     - Total phenols
#     - Flavanoids
#     - Nonflavanoid phenols
#     - Proanthocyanins
#     - Color intensity
#     - Hue
#     - OD280/OD315 of diluted wines
#     - Proline
#     - class:
#         - class_0
#         - class_1
#         - class_2
wine = load_wine()
XTrain, XTest, yTrain, yTest = train_test_split(
    wine.data,
    wine.target,
    random_state=42,
)
df = pd.DataFrame(XTrain, columns=wine.feature_names)
print(df)
colorMap = {
    0 : '#84ffc9',
    1 : '#aab2ff',
    2 : '#eca0ff'
}
colorList = [colorMap[label] for label in yTrain]
generate = scatter_matrix(
    df,
    c = colorList,
    s = 20,
    alpha=0.8,
    figsize=(8,6),
    grid=True,
    diagonal='kde',
    density_kwds= {'linewidth': 3},
    marker='o'
)

print(wine.keys())

knn = KNeighborsClassifier(n_neighbors=1, p=1)
knn.fit(XTrain,yTrain)
yPrediction = knn.predict(XTest)
accuracy = np.mean(yPrediction == yTest)
correctPrediction = accuracy * len(yTest)
incorrectPrediction = len(yTest) - correctPrediction

print(yPrediction)
print(yTest)
print(yPrediction == yTest)
print(correctPrediction)
print(incorrectPrediction)
print(str(np.mean(yPrediction == yTest) * 100) + "%")
plt.show()