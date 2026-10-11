"""Fit bounded models with one job and exercise invalid input."""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_info, threadpool_limits

x = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 3.0, 5.0, 7.0])
model = LinearRegression(n_jobs=1).fit(x, y)
assert np.allclose(model.predict([[4.0]]), [9.0])
assert np.allclose(StandardScaler().fit_transform(x).mean(axis=0), [0.0])
tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, [0, 0, 1, 1])
assert tree.predict([[0.5], [2.5]]).tolist() == [0, 1]
try:
    model.predict([[1.0, 2.0]])
except ValueError:
    pass
else:
    raise AssertionError("invalid feature count accepted")
blas = next(item for item in threadpool_info() if item["internal_api"] == "openblas")
assert blas["version"] == "0.3.31"
assert blas["num_threads"] >= 1
with threadpool_limits(limits=1, user_api="blas"):
    assert next(item for item in threadpool_info() if item["internal_api"] == "openblas")["num_threads"] == 1
print("scikit-learn: estimators, rejection and real OpenBLAS controls passed")
