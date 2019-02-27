import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SDTMMapper=stomioka",
    version="0.0.1",
    author="Sam Tomioka",
    author_email="stomioka@gmail.com",
    description="SDTMMapper for SDTM Mapping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stomioka/sdtm_mapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System  :: OS Independent",
    ],
)