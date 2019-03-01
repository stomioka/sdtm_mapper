import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sdtm_mapper",
    version="0.3.7",
    author="Sam Tomioka",
    author_email="stomioka@gmail.com",
    description="CDISC SDTM Mapping Tool",
    keywords='SDTM, CDISC, SAS, SAS7BDAT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='LICENSE',
    include_package_data=True,
    install_requires=['numpy','pandas','pathlib','sas7bdat','boto3','botocore','sklearn','keras'],
    url='https://github.com/stomioka/sdtm_mapper',
    packages=setuptools.find_packages(exclude=('tests','doc','docs','images','poc','SDTMMapper','train_data','tutorials')),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)