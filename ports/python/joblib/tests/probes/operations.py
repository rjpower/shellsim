"""Use serial jobs and persist a small object in the virtual filesystem."""

import joblib

assert joblib.Parallel(n_jobs=1)(joblib.delayed(pow)(value, 2) for value in range(4)) == [0, 1, 4, 9]
path = "/tmp/port-joblib.pkl"
joblib.dump({"values": [1, 2]}, path)
assert joblib.load(path) == {"values": [1, 2]}
print("joblib: serial jobs and virtual file roundtrip passed")
