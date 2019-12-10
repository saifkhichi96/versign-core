import os
import subprocess
from shutil import rmtree

import cv2

from segment import extract_from_check
from train_test import test





def verify_cheque(user_id, cheque, root_dir=""):
    return verify_signature(user_id, extract_from_check(cheque, root_dir + root_dir + "db/models/tree.pkl", (150, 220)))


def verify_signature(user_id, signature, root_dir=""):
    temp_dir = root_dir + 'db/temp/' + user_id
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    cv2.imwrite(temp_dir + "Q001.png", signature)

    # Extract features from questioned signature
    save_dir = root_dir + 'db/features/'
    model = 'db/models/sabourin/signet.pth'
    results = test(temp_dir, model, save_dir)
    rmtree(temp_dir)

    if results is None:
        return None
    else:
        return results[1][0]
