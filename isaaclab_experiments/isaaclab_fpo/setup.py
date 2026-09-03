from setuptools import find_packages, setup

setup(
    name="isaaclab_fpo",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch~=2.5.1",
        "torchvision~=0.20.1",
        "numpy~=1.26.4",
        "GitPython~=3.1.46",
        "onnx~=1.20.1",
        "viser~=1.0.24",
    ],
)
