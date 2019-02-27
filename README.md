# SDTMMapper
Sam Tomioka

Feb 2019

## About:

**SDTMMapper** is a package to assist creation of CDISC SDTM mapping specifications with Python. This can be used for following tasks.

1. Generates a empty specifications for training data from a user provided SAS dataset. This empty specification will contain SAS dataset attributes.  You don't need to use `Proc Contents` in SAS to do this!
2. Run models to generate a mapping specifications
3. Generates your own mapping models using your data.

The first version comes with **three pre-trained models** ([download from here](https://github.com/stomioka/sdtm_mapper/tree/master/model) for `ADVERSE EVENTS` dataset from CNS clincial trials as well as SDTM IG 3.2 and CDASH IG 1.2 metadata. These are generated with different architectures discussed at several webinars and conference. These are built on feed forward NN with trainable ELMo embedding layer for 34 classes. These are trained on **adverse event** datasets from **Medidata Rave**. Training was done on 18 studies, Validation was done on 3 studies, and Test was done on 1 study.

| Models                 | Parameters | Training Acc | Validation Acc | Test Acc* |
|------------------------|------------|--------------|----------------|----------|
|1. Elmo+sfnn+ae+Model1.h5 | 271,142    |  0.9795        | 0.9800        | 0.9540   |
|2. Elmo+fnn+ae+Model2.h5  | 664,870    | 0.9846      | **1.0000**         | 0.9425   |
|3. Elmo+fnn+ae+Model3.h5  | 594,854    | **0.9966**       | **1.0000**         | **0.9666**   |
**Table 1 - Performance of three models** <br>
\* Macro accuracy account for system variables for 'drop'.

High variance models may be due to addition of CDASH metadata, and probably better to remove them.

Improvement of the task specific model are explored by Peters et.al [1]:

1. Freeze context-independent representations from the pre-trained biLM and concatenate them and $ELMo^{task}_{k}$ and pass that into task RNN.
2. Replacing $h_k$ with $[x_k; ELMo^{task}_{k}]$. Peters et.al [1] has shown improved performance in some tasks such as SNLI and SQuAD by including ELMo at the output of the task RNN.
3. Add a moderate amount of dropout to ELMo.
4. Regularize the ELMo weights by adding $\gamma||w||^2_2$ to the loss function.

These can be considered as future enhancment for other domains that may not perform well.


Here is the architecture of ELMo.

![](images/README-06c97452.png)
**Figure 1** - biLM architecture for ELMo


## References:
1] Peters,M et al. (2018). Deep contextualized word representations

## Notes:
pip installation package is still under development. For now, please clone this git if you want to try. You have to have an environment to use **tensorflow**, **keras**, **scikit-learn** etc.

If you want to contribute for adding more models for different SDTM domains, please join PhUSE ML Project Community.

For any questions, comments, suggestions, or issues, please post them [here](https://github.com/stomioka/sdtm_mapper/issues)

Old reame file is found [here](old_reame.md)


## Tutorial:

1. [Tutorial on how to use SDTMMapper to generate mapping specifications](https://github.com/stomioka/sdtm_mapper/blob/master/tutorials/SDTMMapperTutorial.ipynb)
2. [Train your data using SDTMMapper on Model 1](https://github.com/stomioka/sdtm_mapper/blob/master/tutorials/Build_model_1.ipynb): Note that you need to supply your training data.

## References:
1] Peters,M et al. (2018). Deep contextualized word representations

## Comments, Issues:

For any questions, comments, suggestions, or issues, please post them [here](https://github.com/stomioka/sdtm_mapper/issues)

## Contributions:

Most of the work is done during the weekends or evening. Your contributions are always welcome!

If you want to contribute for adding more models for different SDTM domains, please join [PhUSE ML Project Community](https://www.phusewiki.org/wiki/index.php?title=Machine_Learning_/_Artificial_Intelligence).

## Disclaimer:

The models were build and trained on raw AE datasets from clincial trials conducted by Sunovion Pharmaceuticals. The EDC system we use is Medidata RaveX. The training data contains some e-source data. The performance may not be good for your data.  You can also build your models using SDTMMapper tool and use your custom model for your datasets.
