# VerSign: Easy Signature Verification in Python

```versign``` is a small Python package which can be used to perform verification of offline signatures.

It assumes no prior knowledge of any machine learning tools or machine learning itself, and therefore can be used by ML experts and anyone else who wants to quickly integrate this functionality into their project.

## Getting Started
### Installation
This package requires python 3. Installation can be done with pip:
```
pip install versign
```

You might also need to manually install the following dependencies:
```
pip install git+git://github.com/luizgh/visdom_logger#egg=visdom_logger
pip install git+https://github.com/luizgh/sigver.git
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
It is assumed that only positive samples (i.e. genuine signatures) are available during training, while both genuine and forged signatures are present during testing.

### Write Your First Program with ```VerSign```
```
import os
from versign import VerSign


# Load training data
train_data # folder containing training data (only genuine samples)
x_train = [os.path.join(train_data, f) for f in sorted(os.listdir(train_data))]

# Load test data and labels
test_data # folder containing test data
x_test = [os.path.join(test_data, f) for f in sorted(os.listdir(test_data))]

# Train a writer-dependent model from training data
v = VerSign('models/signet.pth', (150, 220))
v.fit(x_train)

# Predict labels of test data
y_pred = v.predict(x_test)
```

For a more complete example and additional features such as measuring test accuracy if groundtruth is known, see [example.py](./example.py).
