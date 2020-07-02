from setuptools import setup, find_packages
import sys

with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(
    name="watsor",
    version="1.0.3",
    author="Alexander Smirnov",
    author_email="aliaksandr.smirnou@gmail.com",
    description="Object detection for video surveillance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/asmirnou/watsor",
    keywords="object person detection video IP camera realtime stream ffmpeg RTSP surveillance",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'PyYaml',
        'Pillow',
        'cerberus',
        'numpy',
        'scipy',
        'opencv-python-headless',
        'shapely',
        'werkzeug',
        'paho-mqtt',
    ],
    extras_require={
        'coral': [
            'edgetpu',
        ],
        'cuda': [
            'pycuda',
            'tensorrt',
        ],
        'cpu': [
            'tensorflow',
        ],
        'lite': [
            'tflite_runtime',
        ],
        'dev': [
            'coverage',
        ]
    },
)
