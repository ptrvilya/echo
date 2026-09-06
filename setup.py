from importlib.metadata import PackageNotFoundError, distribution

from setuptools import find_packages, setup

version = "0.0.1"

# Check if any of OpenCV packages is installed, if not, install the first one
possible_opencv_packages = (
    "opencv-python-headless",
    "opencv-python",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
selected_opencv_package = possible_opencv_packages[0]
for package in possible_opencv_packages:
    try:
        distribution(package)
        selected_opencv_package = package
        break
    except PackageNotFoundError:
        pass

requirements = [
    "numpy",
    "torch",
    "matplotlib",
    "scipy",
    "loguru",
    selected_opencv_package,
]
setup(
    name="echo",
    version=version,
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
)
