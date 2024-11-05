import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)
os.environ['OPENBLAS_NUM_THREADS'] = str(1)
