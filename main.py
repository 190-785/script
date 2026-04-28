import importlib.util
import subprocess
import sys


PACKAGES = [
    "numpy",
    "tensorflow",
    "torch",
    "transformers",
    "sentencepiece",
    "pillow",
    "requests",
    "matplotlib",
    "scikit-learn",
]


def install_package(package_name):
    print(f"Installing {package_name}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name]
        )
        print(f"  {package_name}: installed")
        return True
    except subprocess.CalledProcessError as error:
        print(f"  {package_name}: install failed ({error.returncode})")
        return False


def print_result(label, value):
    print(f"  {label}: {value}")


def run_check(name, check_fn):
    print(f"\n{name}")
    try:
        result = check_fn()
        print_result("status", "PASS")
        if result is not None:
            print_result("output", result)
        return True
    except Exception as error:
        print_result("status", "FAIL")
        print_result("error", error)
        return False


def main():
    print("\nInstalling required libraries...\n")
    install_results = {package: install_package(package) for package in PACKAGES}

    print("\nImport and sanity checks...\n")

    checks = []

    def numpy_check():
        import numpy as np

        arr = np.array([1, 2, 3])
        return f"{arr.tolist()} -> {(arr * 2).tolist()}"

    checks.append(("NumPy", numpy_check))

    def tensorflow_check():
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense

        model = Sequential([Dense(1, input_shape=(2,))])
        sample = tf.constant([[1.0, 2.0]])
        output = model(sample)
        return f"version={tf.__version__}, output_shape={tuple(output.shape)}"

    checks.append(("TensorFlow", tensorflow_check))

    def torch_check():
        import torch
        import torch.nn as nn

        layer = nn.Linear(2, 1)
        sample = torch.tensor([[1.0, 2.0]])
        output = layer(sample)
        return f"version={torch.__version__}, output_shape={tuple(output.shape)}"

    checks.append(("PyTorch", torch_check))

    def transformers_check():
        import transformers
        from transformers import AutoConfig
        from transformers import (
            BertForSequenceClassification,
            BertTokenizer,
            GPT2LMHeadModel,
            GPT2Tokenizer,
            pipeline,
        )

        config = AutoConfig.for_model("bert")
        imported_symbols = [
            pipeline.__name__,
            BertTokenizer.__name__,
            BertForSequenceClassification.__name__,
            GPT2Tokenizer.__name__,
            GPT2LMHeadModel.__name__,
        ]
        return (
            f"version={transformers.__version__}, config={config.model_type}, "
            f"imports={', '.join(imported_symbols)}"
        )

    checks.append(("Transformers", transformers_check))

    def transformers_download_check():
        from transformers import (
            BertForSequenceClassification,
            BertTokenizer,
            GPT2LMHeadModel,
            GPT2Tokenizer,
            pipeline,
        )

        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        bert_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased"
        )
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        text_pipeline = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

        bert_vocab_size = len(bert_tokenizer)
        gpt2_vocab_size = len(gpt2_tokenizer)
        pipeline_output = text_pipeline("Hello world", max_new_tokens=5, num_return_sequences=1)

        return (
            "downloaded="
            f"bert_tokenizer:{bert_tokenizer.__class__.__name__}, "
            f"bert_model:{bert_model.__class__.__name__}, "
            f"gpt2_tokenizer:{gpt2_tokenizer.__class__.__name__}, "
            f"gpt2_model:{gpt2_model.__class__.__name__}, "
            f"pipeline:{text_pipeline.__class__.__name__}, "
            f"bert_vocab={bert_vocab_size}, gpt2_vocab={gpt2_vocab_size}, "
            f"sample_output={pipeline_output[0]['generated_text'][:80]}"
        )

    checks.append(("Transformers downloads", transformers_download_check))

    def sentencepiece_check():
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor()
        return f"processor_created={processor is not None}"

    checks.append(("SentencePiece", sentencepiece_check))

    def pillow_check():
        from PIL import Image

        image = Image.new("RGB", (2, 2), color="white")
        return f"mode={image.mode}, size={image.size}"

    checks.append(("Pillow", pillow_check))

    def requests_check():
        import requests

        return f"version={requests.__version__}, session_ok={requests.Session() is not None}"

    checks.append(("Requests", requests_check))

    def matplotlib_check():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [2, 4, 6])
        return f"backend={matplotlib.get_backend()}, figure={fig.__class__.__name__}"

    checks.append(("Matplotlib", matplotlib_check))

    def sklearn_check():
        from sklearn.model_selection import train_test_split

        x = [[1], [2], [3], [4]]
        y = [0, 0, 1, 1]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.5, random_state=42
        )
        return f"train={len(x_train)}, test={len(x_test)}, labels={y_train + y_test}"

    checks.append(("Scikit-learn", sklearn_check))

    results = {}
    for name, check_fn in checks:
        results[name] = run_check(name, check_fn)

    print("\nSummary\n")
    for package_name, installed in install_results.items():
        print(f"  install {package_name}: {'PASS' if installed else 'FAIL'}")

    for name, passed in results.items():
        print(f"  verify {name}: {'PASS' if passed else 'FAIL'}")

    if all(install_results.values()) and all(results.values()):
        print("\nAll libraries installed and verified successfully.\n")
    else:
        print("\nSome installs or checks failed. Review the summary above.\n")


if __name__ == "__main__":
    main()
