# SDTM Mapper

Clinical trial data are captured through many different systems in various ways in various data models by the CRO’s study builders and vendor database programmers. Regardless of the efforts to standardize the CRFs and third-party data, each study has some study specific requirements. Furthermore, the study builders are not so familiar with our standards and we often have to deal with the variabilities in the metadata of the same type of data we collect across many studies. We are also getting in-license products that do not follow our internal data standard model and we often cannot avoid the non-standardized data.

With the effort of standardizing the EDC, we have seen up to 40 % of SDTM programs can be recycled for in-house studies, but to improve our efficiency, we have started exploring alternate approach, which is to utlize Machine Learning in SDTM mapping.

We have initiated our pilot of SDTM mapping using the NLP and supervised machine learning approach, leveraging the legacy data which includes studies from Phase 1 to Phase 3 across three different CROs and our inhouse studies and from two EDC systems (Medidata and Inform) and various other third vendors as well as the CDISC standards metadata, CDASH IG Ver 2 and SDTM IG Ver 3.2. Our pilot trial demonstrated that we can map our AE domains with ML with 0.98 accuracy (95%CI 0.92 - 0.98) with Cohen’s kappa of 0.89.

In this paper, we will discuss high level overview of how our proto type solution works and our future visions. We will evaluate the performance of ML and the deep learning in the SDTM mapping pilot trial. We intend to make this application as an open source after we complete building models for several other domains.

Keywords:  CDISC, SDTM, Machine Learning, NLP, Text Mining, AI
