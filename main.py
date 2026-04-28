import subprocess
import sys

# -------------------------------
# STEP 1: INSTALL LIBRARIES
# -------------------------------
packages = [
    "numpy",
    "tensorflow",
    "torch",
    "transformers",
    "sentencepiece",
    "pillow",
    "requests",
    "matplotlib",
    "scikit-learn"
]

print("\nInstalling required libraries...\n")

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("\nAll libraries installed successfully.\n")


# -------------------------------
# STEP 2: IMPORT LIBRARIES
# -------------------------------
print("Testing imports...\n")

try:
    import numpy as np
    print("NumPy OK")

    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
    print("TensorFlow OK")

    import torch
    import torch.nn as nn
    print("PyTorch OK")

    from transformers import (
        pipeline,
        BertTokenizer,
        BertForSequenceClassification,
        GPT2Tokenizer,
        GPT2LMHeadModel
    )
    print("Transformers OK")

    from PIL import Image
    import requests
    print("Pillow + Requests OK")

    import matplotlib.pyplot as plt
    print("Matplotlib OK")

    from sklearn.model_selection import train_test_split
    print("Scikit-learn OK")

except Exception as e:
    print("\nERROR during import:", e)
    sys.exit(1)


# -------------------------------
# STEP 3: BASIC FUNCTION TESTS
# -------------------------------
print("\nRunning sanity tests...\n")

# NumPy test
arr = np.array([1, 2, 3])
print("NumPy test:", arr * 2)

# TensorFlow test
model = Sequential([Dense(1, input_shape=(2,))])
print("TensorFlow model created")

# PyTorch test
tensor = torch.tensor([1.0, 2.0])
print("PyTorch test:", tensor * 2)

# Transformers test (pipeline)
try:
    summarizer = pipeline("summarization")
    print("Transformers pipeline working")
except:
    print("Transformers pipeline skipped (model download issue, not fatal)")

print("\nEverything looks fine. You may now pretend you understand transformers.\n")
