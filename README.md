# VerSign: Easy Signature Verification in Python

```versign``` is a small Python package which can be used to perform verification of offline signatures.

It assumes no prior knowledge of any machine learning tools or machine learning itself, and therefore can be used by ML experts and anyone else who wants to quickly integrate this functionality into their project.

## Getting Started
### Requirements
```versign``` relies on pre-trained models made available by [Hafemann](https://github.com/luizgh) under the ```sigver``` project. Head over to this [repository](https://github.com/luizgh/sigver) and perform the steps under **Installation** heading there.

### Installation
This package requires python 3. Installation can be done with pip:
```
pip install versign
```

Installation inside a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) is recommended.

### Download Trained Models
Before you can get started with, there is one more step you need to complete. ```versign``` comes with some pre-trained models which give it its magic.

Download the compressed models [here](https://drive.google.com/file/d/1qPri1_aWoZKu_EErq6xW_AD9EoCe2fb3/view?usp=sharing), and extract them to ```models/``` directory in your project root. Your project directory should look something like this:
```
_ $PROJECT_ROOT
 |__ models/
 |   |__ signet.pth
 |   |__ versign_segment.pkl
 |__ ...
```

### Organise Your Dataset
It is assumed that only positive samples (i.e. genuine signatures) are available during training, while both genuine and forged signatures can be present in the test data.

In general, your dataset should be structured something like below.
```
_ $PROJECT_ROOT
 |__ models/
 |__ data/
 |   |__ train/
 |   |   |__ 001/
 |   |   |   |__ R01.png
 |   |   |   |__ R02.png
 |   |   |   |__ ...
 |   |   |__ 002/
 |   |       |__ ...
 |   |__ test/
 |       |__ 001/
 |       |   |__ Q01.png
 |       |   |__ Q02.png
 |       |   |__ ...
 |       |__ 002/
 |           |__ ...
 |__ ...
```
Here, ```Ref/``` folder contains your training data, with each sub-folder representing one person. In each of the sub-folders in ```Ref/``` folder, there are images of only genuine signatures of that user.

Similarly, the ```Questioned/``` folder contains your test data. The sub-folders in this folder should be same as those in the training folder, except that they can contain both positive and negative signature samples.

### Write Your First Program with ```VerSign```
```
import os

import joblib
import torch

from sigver.featurelearning.models import SigNet
from versign import VerSign


# Define paths to your data
data_path = 'data/'
train_path = data_path + 'train/'   # path to reference signatures
test_path = data_path + 'test/'     # path to questioned signatures
temp_path = data_path + 'temp/'     # temp path where extracted features will be saved
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

# Load models
state_dict = torch.load('models/signet.pth')[0]
net = SigNet().eval()
net.load_state_dict(state_dict)

clf = joblib.load('models/versign_segment.pkl')
v = VerSign(input_size=(150, 220), extraction_model=net, segmentation_model=clf)

# Learn from genuine signatures
v.train_all(train_path, temp_path)

# Classify your test data
results = v.test_all(test_path, temp_path)

# Print out results
for y_test in results:
    print(y_test)

# Cleanup temp files
shutil.rmtree(temp_path) # comment this line if you want to keep extracted features
```

For a more complete example and additional features such as measuring test accuracy if groundtruth is known, see the [example.py](https://github.com/saifkhichi96/versign-core/blob/master/example.py).