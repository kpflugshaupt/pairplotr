import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "pairplotr",
    version = "1.2.3.1",
    author = "Christopher Shymansky",
    author_email = "CMShymansky@gmail.com",
    description = ("Pairplotr is a Python library used to graph combinations " \
                   "of numerical and categorical data in a pair plot"),
    license = "OSI Approved :: Apache Software License",
    keywords = "scikit-learn pandas data visualization pairplot",
    url = "http://packages.python.org/pairplotr",
    packages=['pairplotr'],
    long_description=read('README.md'),
    install_requires=[
            'seaborn',
            'numpy',
            'matplotlib',
            'pandas'
        ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
    ],
)
