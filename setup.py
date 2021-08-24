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
    python_requires=">=3.4",
    entry_points={
        "console_scripts": [
            "tspyro=tspyro.__main__:main",
        ]
    },
    setup_requires=["setuptools_scm"],
    install_requires=[
        "tskit>=0.3.0",
        "flake8",
        "numpy",
        "networkx",
        "pyro-ppl>=1.7",
        "pytest",
        "pandas",
    ],
    project_urls={
        "Source": "https://github.com/awohns/tspyro",
        "Bug Reports": "https://github.com/awohns/tspyro/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
    ],
    use_scm_version={"write_to": "tspyro/_version.py"},
)
