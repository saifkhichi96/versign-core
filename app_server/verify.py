#!/usr/local/bin/env python
import json
import os
import sys

import numpy as np

from ... import verify_signature

# Read input data from JSON
fn = "../../../versign-core/src/app/verify_request.json"
fo = open(fn, "r")
payload = json.loads(fo.read())
fo.close()
os.remove(fn)

# Get customer ID
user = payload['customerId']

# Extract all four reference signatures
refSign = payload['questionedSignature']
pixelData = np.array(refSign['pixelData']).astype('uint8')
width = refSign['width']
height = refSign['height']

a = np.reshape(pixelData, (height, width))
print
verify_signature(user, a, "../../../versign-core/")
