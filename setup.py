from setuptools import setup, find_packages


with open("evolutionary/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")


setup(
    name='evolutionary',
    version=version,
    author='Reuben Brasher and Emma Sawin',
    install_requires=["numpy~=1.24", "scikit-learn~=1.2"],
    packages=find_packages(),
    include_package_data=True
)
