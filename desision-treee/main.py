from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
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

# ジニ不純度を指標に決定木のインスタンスを生成
tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)

# # 決定木のモデルを訓練データに適合させる
tree_model.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
Y_combined = np.hstack((y_train, y_test))
plt.figure()
# plot_decision_regions(X_combined, Y_combined, classifier=tree_model,
#                       test_idx=range(105, 150))
plot_decision_regions(X_combined, Y_combined, clf=tree_model,
                      X_highlight=X_test)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.tight_layout()

# 訓練後の決定木モデルの可視化
from sklearn import tree
plt.figure()
feature_names = ['Petal length', 'Petal width']
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()