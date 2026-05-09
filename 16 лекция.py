# SciKit-Learn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
print(iris.head())

# print(type(iris))

# print(type(iris.values))

# print(iris.values.shape)

# print(iris.columns)

# print(iris.index)

# sns.pairplot(iris, hue="species")

# Строки - оразцы - отдельный обьект (sample)
# Столбцы - признаки (feature)
# Матрица признаков [число образцов на число признаков]
# Целевой массив (target, lable) [1 на число образцов]

X_iris = iris.drop("species", axis=1)
# print(X_iris)

y_iris = iris["species"]
# print(y_iris)

# 1. Выбирается класс модели
# 2. Выбираются гиперпараметры модели
# 3. На основе данных создается матрица признаков и целевой вектор
# 4. Обучение модели fit()
# 5. Обученная модель применяется к новым данным
#    5.1. Обучение с учителем - predict()
#    5.2. Обучение без учителя - predict() или transform()

# С учителем. Регрессия. Линейная регрессия


# 1. Выбирается класс модели
from sklearn.linear_model import LinearRegression

# 2. Выбираются гиперпараметры модели
model = LinearRegression(fit_intercept=True)

# 3. На основе данных создается матрица признаков и целевой вектор

x = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()

# 4. Обучение модели fit()
reg = model.fit(x[:, np.newaxis], y)

# plt.scatter(x, y)

# 5. Обученная модель применяется к новым данным
#    5.1. Обучение с учителем - predict()
#    5.2. Обучение без учителя - predict() или transform()

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, None])

# plt.scatter(xfit, yfit)
# plt.plot(xfit, yfit, "r")

# plt.plot(xfit, yfit * reg.coef_ + reg.intercept_, "k")

# y = kx + b

# plt.show()
# exit()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(7), LinearRegression())
reg = model.fit(x[:, np.newaxis], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, None])

# plt.scatter(x, y)
# plt.plot(xfit, yfit, "r")

# plt.show()
# exit()

# Классификация. Логистическая регрессия

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()

x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

# plt.scatter(x_0, y_0, color="red", alpha=0.5)
# plt.scatter(x_1, y_1, color="green", alpha=0.5)

x_00 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
x_11 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()

# plt.scatter(x_00, np.full(50, 1), color="red", alpha=0.5)
# plt.scatter(x_11, np.full(50, 2), color="red", alpha=0.5)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

x = iris[iris["species"] != "virginica"].iloc[:, 0].to_numpy()
print(x.shape)
y = iris[iris["species"] != "virginica"].iloc[:, 4].to_numpy()
print(y.shape)
print(y)

model.fit(x[:, None], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict_proba(xfit[:, None])

# print(yfit)

# plt.plot(xfit, 1 + 4 * yfit[:, 1], "green")

# plt.plot(xfit, 1 + 4 * yfit[:, 0], "red")

# plt.show()
# exit()

# Деревья решений

from sklearn.tree import DecisionTreeClassifier

x = iris[iris["species"] != "virginica"].iloc[:, 0:2].to_numpy()
y = iris[iris["species"] != "virginica"].iloc[:, 4].to_numpy()
y1 = np.full(50, 1)
y2 = np.full(50, 2)
print(y1)
print(type(y1))


y = np.ravel([y1, y2])

print(x)
print(y)

tree = DecisionTreeClassifier()
tree.fit(x, y)

print(np.c_[[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]])

print(np.ravel([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]))

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 1000),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 1000),
)

Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ax = plt.gca()

# ax.contour(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

# plt.show()

# Метод опорныйх векторов

from sklearn.svm import SVC

x = iris[iris["species"] != "virginica"].iloc[:, 0:2].to_numpy()
y = iris[iris["species"] != "virginica"].iloc[:, 4].to_numpy()

y1 = np.full(50, 1)
y2 = np.full(50, 2)
y = np.ravel([y1, y2])

print(x)
print(y)

model = SVC(kernel="linear", C=1e10)

model.fit(x, y)

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()

x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

# plt.scatter(x_0, y_0, color='red', alpha=0.5)
# plt.scatter(x_1, y_1, color='green', alpha=0.5)

# Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# ax = plt.gca()
# ax.contour(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])


# Наивная Баесовская классификация

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 1000),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 1000),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()

x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

# plt.scatter(x_0, y_0, color='red', alpha=0.5)
# plt.scatter(x_1, y_1, color='green', alpha=0.5)

# ax = plt.gca()
# ax.contour(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

x_m = model.theta_[0]
x_var = model.var_[0]

y_m = model.theta_[1]
y_var = model.var_[1]

z1 = (
    1
    / (2 * np.pi * (x_var[0] * y_var[1]) ** 0.5)
    * np.exp(
        -((xx - x_m[0]) ** 2) / (2 * x_var[0]) - (yy - x_m[1]) ** 2 / (2 * x_var[1])
    )
)

# ax.contour(xx, yy, z1)


z2 = (
    1
    / (2 * np.pi * (y_var[0] * y_var[1]) ** 0.5)
    * np.exp(
        -((xx - y_m[0]) ** 2) / (2 * y_var[0]) - (yy - y_m[1]) ** 2 / (2 * y_var[1])
    )
)

# ax.contour(xx, yy, z2)


# ax = plt.axes(projection='3d')

# ax.contour3D(xx, yy, z1, 50)
# ax.contour3D(xx, yy, z2, 50)

# plt.show()

# kNN

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 1000),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 1000),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

ax = plt.gca()
ax.contour(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

plt.show()
