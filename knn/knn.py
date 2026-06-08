from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions


# データ準備
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

# 特徴量(X_train, X_test)の正規化
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# KNNのインスタンスを生成
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
# KNNモデルを訓練データを暗記させる
knn.fit(X_train_std, y_train)


# 訓練データとテストデータを結合
X_combined = np.vstack((X_train_std, X_test_std))
Y_combined = np.hstack((y_train, y_test))

# 結果の表示
plot_decision_regions(X_combined, Y_combined, clf=knn,
                      X_highlight=X_test_std)
plt.xlabel('petal length [standardised]')
plt.ylabel('petal width [standardised]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()