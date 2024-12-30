from distutils.core import setup
import py2exe
from setuptools import setup, find_packages

setup(
    console=['main.py'],
    options={"py2exe": {"bundle_files": 1, "include_files": ["meat_icon.ico", "resources/"]}},name='YourProjectName',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here, for example:
        'numpy',
        'pandas',
        'py2exe',
        'joblib',
'pandas',
'matplotlib',
'scikit-learn',
'tk',
'tensorflow'
    ],

)
