# Scikit-learn

Build Scikit-learn 1.8.0 with target NumPy headers and SciPy's exported Cython
declarations. Host Python/Cython generate code; guest libraries provide runtime
execution. Tag the upstream Python package for runtime installation.
Qualify one-job linear regression, scaling, a bounded decision tree and real
OpenBLAS inspection/limits through Threadpoolctl.
OpenMP is optional upstream and unavailable in this target build. Process-worker
backends and other estimators remain outside the declared qualification.
