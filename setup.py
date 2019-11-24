import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="draw_gym-mthorrell", # Replace with your own username
    version="0.0.1",
    author="mthorrell",
    author_email="",
    description="RL environment for drawing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mthorrell/draw_gym",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5.2',
)