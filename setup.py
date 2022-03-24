from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="tspyro",
    author="Anthony Wilder Wohns, Fritz Obermeyer, Eli Bingham, Yan Wong",
    author_email="awohns@gmail.com",
    description="Infer geography and time in a tree sequence",
    long_description=long_description,
    packages=["tspyro"],
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    setup_requires=["setuptools_scm"],
    install_requires=[
        "networkx",
        "numpy",
        "pandas",
        "pyro-ppl>=1.7",
        "python-louvain",
        "torch-scatter",  # from https://data.pyg.org/whl/torch-1.11.0+cpu.htm
        "tsdate",
        "tskit",
    ],
    extras_require={
        "test": [
            "black",
            "flake8",
            "mypy>=0.812",
            "pyslim",
            "pytest",
        ],
    },
    project_urls={
        "Source": "https://github.com/awohns/tspyro",
        "Bug Reports": "https://github.com/awohns/tspyro/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    use_scm_version={"write_to": "tspyro/_version.py"},
)
