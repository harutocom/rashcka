from sklearn.ensemble import RandomForestClassifier
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

# ランダムフォレストのインスタンスを作成
forest = RandomForestClassifier(n_estimators=1000, random_state=1, n_jobs=2)
# ランダムフォレストモデルを訓練データに適合させる
forest.fit(X_train, y_train)

# 訓練データとテストデータを結合
X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((y_train, y_test))

# 結果の表示
plot_decision_regions(X_combined, Y_combined, clf=forest,
                      X_highlight=X_test)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()