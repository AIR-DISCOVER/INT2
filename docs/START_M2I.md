**Step 1:** clone the M2I codebase, and configure it's environment
```
cd int_benchmark
git clone git@github.com:Tsinghua-MARS-Lab/M2I.git

# After that, please configure the environment according to its requirements
```

**Step 2:** preprocess INT2 Dataset to M2I needs format.

```
# Edit python data_format_preprocess_m2i.py, modify the path in the main function to the path you want, default does not need to be modified. The process takes about 12 hours.

python data_format_preprocess_m2i.py
```

**Step 3:** Runing configuration.
```
cp dataset_int2.py M2I/src
cd M2I/src
cython -a utils_cython.pyx && python setup.py build_ext --inplace

# Modify run.py line 427 and line 465:  from dataset_waymo import Dataset   =======>  from dataset_int2 import Dataset
```


**Step 4:** Runing Model.
'''

'''