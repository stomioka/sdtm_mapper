# SDTMMapper
Sam Tomioka
Feb 2019

## About:

SDTMMapper is a package to assist creation of CDISC SDTM mapping specifications with Python. This can be used for following tasks.

1. Generates a empty specifications for training data from a user provided SAS dataset. This empty specification will contain SAS dataset attributes.  You don't need to use `Proc Contents` in SAS to do this!
2. Run models to generate a mapping specifications
3. Generates your own mapping models using your data.


The first version comes with three pre-trained models ([download from here](https://github.com/stomioka/sdtm_mapper)) for `ADVERSE EVENTS` dataset from CNS clincial trials as well as SDTM IG 3.2 and CDASH IG 1.2 metadata. These are built on feed forward NN with ELMo embedding layer for 34 classes.


| Models                 | Parameters | Training Acc | Validation Acc | Test Acc* |
|------------------------|------------|--------------|----------------|----------|
| Elmo+sfnn+ae+Model1.h5 | 271,142    | 0.9229       | 0.9900         | 0.9080   |
| Elmo+fnn+ae+Model2.h5  | 664,870    | 0.9247       | 0.9800         | 0.9195   |
| Elmo+fnn+ae+Model3.h5  | 594,854    | 0.9903       | 0.9906         | 0.9310   |

These are trained on adverse event datasets from Medidata Rave. Training was done on 18 studies, Validation was done on 3 studies, and Test was done on 1 study.
* Macro accuracy account for system variables for 'drop'.

High variance models may be due to addition of CDASH metadata, and probably better to remove them.

This tool will accept pre-traing models from Keras. More models and domains will be added as part of PhUSE Project. 

If you want to contribute for adding more models for different SDTM domains, please join PhUSE ML Project Community.

For any questions, comments, suggestions, or issues, please post them [here](https://github.com/stomioka/sdtm_mapper/issues)


## Tutorial:

[Please find from here](https://github.com/stomioka/sdtm_mapper)


## Contributions:

Most of the work is done during the weekends or evening. Your contributions are always welcome!


## Disclaimer:

The models were build then trained on raw AE datasets from clincial trials conducted by Sunovion Pharmaceuticals. The EDC system we use is Medidata RaveX. The training data contains some e-source data. The performance may not be good for your data.  You can also build your models using SDTMMapper tool and use your custom model for your datasets.

