# Installation
The installation of INT2 is super easy.

**Step 1:** Create an python environment for INT2
```shell
conda create --name int2 python=3.8 -y
conda activate int2
```

**Step 2:** Install the required packages
```shell
pip install -r requirements.txt
```

**Step 3:** Clone the codebase, and compile this codebase: 
```python
# Clone INT2 codebase
git clone https://github.com/BJHYZJ/INT2.git

# Compile the customized CUDA codes in INT2 codebase
python setup.py develop
```

Then you are all set.