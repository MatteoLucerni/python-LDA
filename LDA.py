import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


iris = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    names=["sepal length", "sepal width", "petal length", "petal width", "target"],
)

iris.head()

X = iris.drop("target", axis=1).values
Y = iris["target"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# PCA
pca = PCA(n_components=2)
pc_train = pca.fit_transform(X_train)
pc_test = pca.transform(X_test)

plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.scatter(pc_train[:, 0], pc_train[:, 1], c=Y_train)
plt.scatter(pc_test[:, 0], pc_test[:, 1], c=Y_test, alpha=0.5)
plt.show()

# lr on PCA
lr = LogisticRegression()
lr.fit(pc_train, Y_train)

pred = lr.predict(pc_test)
pred_train = lr.predict(pc_train)

proba = lr.predict_proba(pc_test)
proba_train = lr.predict_proba(pc_train)

acc = accuracy_score(Y_test, pred)
acc_train = accuracy_score(Y_train, pred_train)

loss = log_loss(Y_test, proba)
loss_train = log_loss(Y_train, proba_train)

print(
    f"PCA -> Acc: (test {acc} / train {acc_train}) - Loss: (test {loss} / train {loss_train})"
)

# LDA
lda = LDA(n_components=2)

ld_train = lda.fit_transform(X_train, Y_train)
ld_test = lda.transform(X_test)

plt.xlabel("First discriminant")
plt.ylabel("Second discriminant")
plt.scatter(ld_train[:, 0], ld_train[:, 1], c=Y_train)
plt.scatter(ld_test[:, 0], ld_test[:, 1], c=Y_test, alpha=0.5)
plt.show()

# lr on LDA
lr = LogisticRegression()
lr.fit(ld_train, Y_train)

pred = lr.predict(ld_test)
pred_train = lr.predict(ld_train)

proba = lr.predict_proba(ld_test)
proba_train = lr.predict_proba(ld_train)

acc = accuracy_score(Y_test, pred)
acc_train = accuracy_score(Y_train, pred_train)

loss = log_loss(Y_test, proba)
loss_train = log_loss(Y_train, proba_train)

print(
    f"LDA -> Acc: (test {acc} / train {acc_train}) - Loss: (test {loss} / train {loss_train})"
)
