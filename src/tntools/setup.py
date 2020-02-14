import setuptools

with open("README.md", "r") as f:
    readme = f.read()

setuptools.setup(
    name="tntools",
    version="0.1.0",
    author="Markus Hauru",
    author_email="markus@mhauru.org",
    description="An assortment of tools for developing tensor network algorithms in Python 3.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mhauru/tntools",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    keywords=["tensor networks"],
    install_requires=["scipy>=1.0.0", "pyyaml", "abeliantensors", "ncon"],
    python_requires=">=3.6",
)
