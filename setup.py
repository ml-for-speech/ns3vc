from setuptools import setup, find_packages

with open("README.md", "r") as f:
    longdesc = f.read()
setup(
    name="ns3vc",
    version="1.0.0",
    author="ml-for-speech, mrfakename",
    description="Unofficial pip package for Amphion's NaturalSpeech3 implementation",
    long_description=longdesc,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-for-speech/ns3vc",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "librosa",
        "matplotlib",
        "tqdm",
        "hydra-core",
        "torch",
        "torchvision",
        "torchaudio",
        "tensorboard",
        "onnxruntime",
        "onnx",
        "pyworld",
        "pesq",
        "pystoi",
        "gradio",
        "einops",
        "PyWavelets",
        "cached_path",
    ],
)
