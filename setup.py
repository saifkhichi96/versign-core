import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="versign",
    version="0.0.2",
    author="Saif Khan",
    author_email="saifkhichi96@gmail.com",
    description="Signature verification package for verifying offline signatures using writer-independent features.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saifkhichi96/versign-core",
    packages=setuptools.find_packages(),
    python_requires='>=3',
    install_requires=[
        'joblib',
        'numpy>=1.10.4',
        'opencv-contrib-python',
        'Pillow',
        'scikit-image',
        'scikit-learn>=0.19.0',
        'scipy',
        'torch>=1.0',
        'torchvision>=0.2.1',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
