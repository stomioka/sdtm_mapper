
# NLP and ML based SDTM mapping
## Non-rule based approach

## Background

Clinical trial data are captured through many different systems in various way. Regardless of the effort to standardize the CRFs, eCOA, clinical laboratory data, central ECG, IxRS, PK ... by Sunovion programmers, the CROs and vendor database programmers generate database tables in many different way across studies.

In 2017, Sunovion programmers received 12 clinical trials raw data from one CRO. Across 12 studies, total of 497 forms, 187 unique forms were generated, and 241 datasets were produced. Only 1 % of datasets (n=3) were used consistently without change in variable attributes. This significantly increases the time to produce the programming specifications, programming code, and validation.

Certainly, there is a need for standard metadata for the raw data and we expect that that would bring the consistency level to approximately 40% based on the historical data from Sunovion in-house studies.
In order to further increase the SDTM programming efficiency, machine learning approach is explored.


## Objectives

1. Build a statistical learning model using metadata of the raw datasets and the mapping to auto-map to intermediate SDTM variables
2. Build a ruled based machine learning models to auto-map to intermediate SDTM variables
3. Pilot the use of a neural network using TensorFlow to deal with the outcome that are not expected to be in the normal distribution. TensorFlow which is the open source machine learning framework from Google. TensorFlow is an open-source software library for dataflow programming across a range of tasks released on November 9, 2015. It is a symbolic math library, and is also used for machine learning applications such as neural networks. It is used for both research and production at Google. https://github.com/tensorflow/tensorflow/tree/r1.1
Use cases:https://tensorflow.rstudio.com/learn/gallery.html
4. Pilot the use of greta
https://greta-dev.github.io/greta/

## Method

## NLP techniques used for featuer extractions
1. Stemming - Stemming is a process applied to a single word to derive its root. Many words that are being used in a sentence are often inflected or derived. To standardize our process, we would like to stem such words and end up with only root words. For example, a stemmer will convert the following words "walking", "walked", "walker" to its root word "walk".
In this pilot, I will use Porter's word stemming algorithm which is most frequently used method.

2. Tokenization - Tokens are basically words. This is a process of taking in a piece of text and find out all the unique words in the text. We would get a list of words in the text as the output of tokens. For example, for the sentence "Python NLP is just going great" we have the token list [ "Python", "NLP", is", "just", "going", "great"]. So, as you can see, tokenization involves breaking up the text into words. n-grams method where n=2 is found to be most robust in the SDTM mapping.
3. Bag of Words (BoW) - The Bag of Words model in Text Processing is the process of creating a unique list of words. This model is used as a tool for feature generation. 
Reference: https://ongspxm.github.io/blog/2014/12/bag-of-words-natural-language-processing/

3. Term frequency is
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/91699003abf4fe8bdf861bbce08e73e71acf5fd4)
where t is a word, and d is the variable.

4. Inverse document frequency is
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/ac67bc0f76b5b8e31e842d6b7d28f8949dab7937)

where N: total number of variables, denominator is the number of variables where the word t appears.

For this pilot, weightTfIdf in tm library (ver 0.7-3) is used rather than weightdf since it led to better trained models.

## Conventions used for the intermediate SDTM
1.	One or many raw variable is mapped to one SDTM variable. In case of one raw variable to many SDTM variables, only one SDTM variable is kept, since other SDTM variable(s) can be derived programmatically,
2.	Any raw variables that do not fit into SDTM data model but maybe needed for analysis is padded with a prefix of QNAM_. This is for SUPPQUAL domains,
3.	Partial dates need to be populated in ISO8601 format, so character based raw date (without time) variables are mapped to --DTC/----DTC variables. Time raw variables are mapped to --DTC_TM/----DTC_TM,
4.	Some flag variables in the raw data are useful in the SDTM programming, therefore these will have prefix of F_.  For instance, "same as visit variable" (F_VIS) is not mapped directly but used to get the visit date as an assessment date, 
5.	De-normalized structure is a preferred choice for finding domains in EDC. Therefore, each test name has its own field name. These are labeled as -ORRES_{TEST Code}. For example, VSORRES_HEIGHT. Similarly, measurement units are labeled as -ORRESU_{TEST Code}. 
6.	Any raw variables that are not mapped are labeled as 'drop'. The records are kept in case SDTM programmers choose to use them.


## Generate Training data
Following twenty studies were selected to build a training data. These studies were from Phase 1 to Phase 3 studies build by Sunovion, **Reducted**, and **Reducted** and across multiple indications.Total of 56357 raw variables were identified. Variable attributes and the first 10 observations within each dataset were extracted.

[inventory.sas](Z:\inventory\inventory.sas) was used for metadata extraction.

**The list reducted**

CDISC Standard metadata are also included in the training (You need to have CDISC account to download these)

- [CDASHIG2](https://www.cdisc.org/system/files/gold/eshare/CDASHIGv2.0_MetadataTable.xlsx)
- [SDTMIG](https://www.cdisc.org/system/files/gold/eshare/sdtm-3-2-diff.xls)

## Setup


```R

options(warn=-1)
suppressPackageStartupMessages(library(RTextTools))
                 library(dplyr)
                 library(caret)
                 library(tidytext)
                 library(tidyverse)
                 library(purrr)
                 library(tidyr)
                 library(SnowballC) #to use Porter Stemmer algorithm https://tartarus.org/martin/PorterStemmer/def.txt
                 library(rattle)   # Fancy tree plot
                 library(caTools)
                 library(tidyr)
                 library(readxl)
                 library(tidyr)
                 library(ggplot2)

#plot accuracy/kappa plot for tree 
plot_ak<-function(input,xvar,w, title) {

  r1<-gather(data=input$results[,1:3],key="Metrics", value="Value",-1)
  sd<-gather(data=input$results[,c(1,4,5)],key="Metrics", value="sd",-1)[,c(3)]
  r1<-cbind(r1,sd) 
  pl<-ggplot(data=r1,aes_string(x=xvar,y="Value",col="Metrics"))+geom_line()+
    geom_point()+    geom_errorbar(aes(ymin=Value-sd,ymax=Value+sd),width=w)+
    labs(x=colnames(input$results[1]),y="Accuracy/Kappa (+/1 SD)", title=title)
  rm(r1)
  print(pl)
  
}
plot_ak2<-function(input,xvar,w, title) {
  library(tidyr)
  r1<-gather(data=input$results[,c(1, 4, 5)],key="Metrics", value="Value",-1)
  sd<-gather(data=input$results[,c(1,6,7)],key="Metrics", value="sd",-1)[,c(3)]
  r1<-cbind(r1,sd) 
  pl<-ggplot(data=r1,aes_string(x=xvar,y="Value",col="Metrics"))+geom_line()+
    geom_point()+    geom_errorbar(aes(ymin=Value-sd,ymax=Value+sd),width=w)+
    labs(x=colnames(input$results[1]),y="Accuracy/Kappa (+/1 SD)", title=title)
  rm(r1)
  print(pl)
  
}
# sample call = plot_ak(rf_mod1,"mtry",w=3)

setwd("~/My Documents/r/sdtm_mapping") # work laptop 16 Ram
#setwd("~/R/sdtm_mapping") # home desktop 32 RAM, 6 core
#setwd("D:/My Documents/r/sdtm_mapping") #workstation 48 RAM 12 ore
options(warn=0)
```

    
    Attaching package: 'dplyr'
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    Loading required package: lattice
    Loading required package: ggplot2
    -- Attaching packages --------------------------------------- tidyverse 1.2.1 --
    v tibble  1.4.2     v purrr   0.2.5
    v tidyr   0.8.1     v stringr 1.3.1
    v readr   1.1.1     v forcats 0.3.0
    -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    x dplyr::filter() masks stats::filter()
    x dplyr::lag()    masks stats::lag()
    x purrr::lift()   masks caret::lift()
    
    Attaching package: 'SnowballC'
    
    The following objects are masked from 'package:RTextTools':
    
        getStemLanguages, wordStem
    
    Rattle: A free graphical interface for data science with R.
    Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
    Type 'rattle()' to shake, rattle, and roll your data.
    

## Obtain CDISC CDASH Metadata


```R

#cdash <- "https://www.cdisc.org/system/files/gold/eshare/CDASHIGv2.0_MetadataTable.xlsx"
#download.file(cdash, destfile="./train_data/cdash.xlsx",method="auto")

 cdash <- read_excel("./train_data/CDASHIGv2.0_MetadataTable.xlsx", 
     col_types = c("skip", "text", "skip", 
         "skip", "skip", "text", "text", 
         "text", "text", "text", "skip", 
         "skip", "skip", "text", "skip", 
         "skip", "skip", "skip")) 
 names(cdash)<-make.names(names(cdash))
 cdash<-cdash %>% mutate(text=paste(CDASHIG.Variable, CDASHIG.Variable.Label,DRAFT.CDASHIG.Definition,Question.Text,Prompt)) %>%
   mutate(ID=paste0(Domain,".",CDASHIG.Variable)) %>%
    #Add QNAM_ to SUPP vars
    mutate(sdtm=ifelse(grepl(pattern="^SUPP*",SDTMIG.Target),
                      paste0("QNAM_",CDASHIG.Variable),SDTMIG.Target)) %>%
    #remove CDASH variable with bracket
    subset(!grepl(pattern="[",CDASHIG.Variable,fixed=TRUE)) %>%
    #check<-cdash  %>% subset(grepl(pattern="[",CDASHIG.Variable,fixed=TRUE)) 
    # remove sdtm=na
    subset(toupper(sdtm)!="N/A") %>% 
    mutate(LIBNAME="CDASHIG2") %>%
    mutate(ID=toupper(paste0(LIBNAME,".",Domain,".", CDASHIG.Variable))) %>%

    dplyr::select(c("Domain","ID","SDTMIG.Target","text")) 
 
    #remove variable name (in upper case with at least two repeated) from text because they may confuse
     text<- (gsub("[A-Z]{2,}", "", cdash$text))
      a <-as.matrix(x=text,ncol=1,nrow=nrow(cdash)) %>% as.data.frame()

      cdash<- cbind(cdash[,c(1,2,3)],a) 
      cdash<- cdash[!endsWith(x=cdash$SDTMIG.Target,c("SDTY","ENDY","DY","USUBJID","GRPID","AEREFID","SEQ","SITEID","DOMAIN","FAORRES")),]
    names(cdash)<-c("DataPageName","ID","sdtm","text")
```

## Obtain SDTM IG 3.2 Metadata


```R
#sdtm<-"https://www.cdisc.org/system/files/gold/eshare/sdtm-3-2-excel.xls"
#download.file(sdtm, destfile="./train_data/sdtm.xlsx",method="auto")   
sdtm <- read_excel("./train_data/sdtm-3-2-excel.xls", 
    col_types = c("skip", "skip", "text", 
        "skip", "text", "text", "skip", 
        "skip", "skip", "skip", "skip", 
        "skip", "skip", "skip", "skip", 
        "skip"))
  names(sdtm)<-make.names(names(sdtm)) 
  #drop NA from domain
  sdtm<-sdtm %>% subset(!is.na(Domain.Prefix)) %>% mutate(sdtm=Variable.Name) %>% 
    mutate(LIBNAME="SDTMIG32") %>%
    mutate(ID=toupper(paste0(LIBNAME,".",Domain.Prefix,".", Variable.Name))) %>%
    mutate(text=paste(sdtm,Variable.Label))  %>% 
    dplyr::select(-2,-3,-5)  %>% subset(!sdtm%in%c("USUBJID","SITEID","DOMAIN","AEREFID")) %>% 
      subset(!endsWith(x=sdtm,"STDY"))    %>%
      subset(!endsWith(x=sdtm,c("ENDY")))   %>% 
      subset(!endsWith(x=sdtm,c("DY")))   %>%     
      subset(!endsWith(x=sdtm,c("GRPID")))   %>%  
        subset(!endsWith(x=sdtm,c("SEQ"))) 
    
  names(sdtm)<-c("DataPageName","sdtm","ID","text")
  
  sdtm$sdtm<-gsub(x =sdtm$sdtm , pattern = "AEBDSYCD",replacement ="AESOCCD" )
   sdtm$sdtm<-gsub(x =sdtm$sdtm , pattern = "AEBODSYS",replacement ="AESOC" )
  
  
  
  sdtm<-rbind(sdtm,cdash)  %>% subset(DataPageName=="AE") %>%
        dplyr::select(-1)
  write.csv(sdtm, file="./train_data/sdtm_cdash_meta.csv",row.names=F)
```

## Import the rest of the training data


```R
#options(stringsAsFactors = FALSE)
meta<- read.csv( file="./train_data/rawmeta_st2018.csv", header=T) %>% 
  dplyr::filter(DataPageName=="Adverse Events")
newdata<- read.csv( file="./train_data/testmeta.csv", header=T)

#Training Data & Test Data & New Data

meta1 <- meta %>%  dplyr::filter(DataPageName=="Adverse Events") %>% 
      mutate ( text=paste(NAME, LABEL, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, sep=" ")) %>% 
      mutate ( text2=paste(NAME, LABEL, sep=" ")) %>% 
      mutate ( text3=paste(NAME, sep=" ")) %>%
      dplyr::filter(!MEMNAME %in% c("AE_RAW","CHANGE_AE")) %>% 
      dplyr::select(-(10:19)) %>%
      mutate(ID=toupper(paste0(LIBNAME,".",MEMNAME,".", NAME))) %>% 
      mutate(ID2="") %>% 
      mutate(sdtm=as.factor(toupper(sdtm))) %>%
      dplyr::select(c("ID","text","text2", "text3","DataPageName","TYPE","LENGTH","ID2","sdtm")) 

    #remove underscore
      #meta1$uflg2<-grepl("_", meta1$text3)*1
      meta1$text<-gsub("_"," ", meta1$text)
      meta1$text3<-gsub("_"," ", meta1$text3)
      meta1$text2<-gsub("_"," ", meta1$text2)  
 

##New Data
#test_meta1 <- newdata %>%  dplyr::filter(DataPageName=="Adverse Events") %>% 
#      mutate ( text=paste(NAME, LABEL, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, sep=" ")) %>% 
#      mutate ( text2=paste(NAME, LABEL, sep=" ")) %>% 
#      mutate ( text3=paste(NAME, sep=" ")) %>%
#      dplyr::filter(!MEMNAME %in% c("AE_RAW","CHANGE_AE")) %>% 
#      dplyr::select(-(10:19)) %>%
#      mutate(ID=toupper(paste0(LIBNAME,".",MEMNAME,".", NAME))) %>% 
#      mutate(ID2="") %>% 
#      mutate(sdtm=as.factor(toupper(sdtm))) %>%
#      dplyr::select(c("ID","text","text2", "text3","DataPageName","TYPE","LENGTH","ID2","sdtm")) 
#
#    #remove underscore
#      #meta1$uflg2<-grepl("_", meta1$text3)*1
#      test_meta1$text<-gsub("_"," ", test_meta1$text)
#      test_meta1$text3<-gsub("_"," ", test_meta1$text3)
#      test_meta1$text2<-gsub("_"," ", test_meta1$text2)  
            
write.csv(meta1, file="./train_data/ae_meta_simple.csv",row.names=F)
#write.csv(test_meta1, file="./train_data/newdata.csv",row.names=F)

rm(meta, meta1, a, cdash, sdtm)
```

## Manipulate training, validation, and test data for machine leraning

To do a quick pilot of the methodology, AE domain was chosen.
Initially the training dataset was prepared with consideration of variable labels, and the first 10 row values of each variable, but it caused significant increase in the use of CPU and RAM as well as the time to build each model but did not improve the mapping accuracy, therefore, it was decided to use only variable labels and the variable name along with the data type. Natural language found in these variables were converted into tokens using n-grams method and each token was generalized with Porter's word stemming algorithm.


The final training dataset for AE domain has `r nrow(meta1)` raw variables. Because each raw dataset contains some variables from the system for drop, sampling needs to be considered for unbalanced distribution of the class.

Note: training dataset must be accurate. Same raw variable cannot map to more than one SDTM variable

## Feature Engineering


```R
suppressMessages(
ae_meta <-  read_csv(paste0("./train_data/ae_meta_simple.csv")) %>% 
                  mutate(text=toupper(text2)) %>% 
                  dplyr::select(c("ID", "text", "sdtm"))
)
suppressMessages(cdisc_meta <-read_csv(paste0("./train_data/sdtm_cdash_meta.csv"))) 

ae_meta <-rbind(ae_meta,cdisc_meta)
suppressMessages(write.csv(ae_meta, file="./train_data/training_data.csv",row.names=F))
rm(cdisc_meta)
suppressMessages(ae_meta <-  read_csv(paste0("./train_data/training_data.csv")))

      #Distribution
      ae_meta %>%
      ggplot(aes(sdtm)) +
      geom_bar()+labs(title="Fig 1: distribution of class")

      ae_meta$sdtm <- factor(ae_meta$sdtm)
options(warn=-1)      

      data_dtm2 <-  map_df(1:2, ~ unnest_tokens(ae_meta, word, text, token = "ngrams",  n = 1)) %>%
                    anti_join(stop_words, by="word") %>% #remove common stop words that are uninformative
                    mutate(word=SnowballC::wordStem(word))%>% #remove suffix using Porter's word stemming algorithm
                    count(ID,word)       %>%
      
      #derive the tf_idf and generate document term matrix
      bind_tf_idf(word, ID, n)  %>%  cast_dtm    (document=ID, term=word, value=n, weighting=tm::weightTfIdf) #adjust for the fact that some words appear more
  
     meta <- tibble(ID = as.character(dimnames(data_dtm2)[[1]])) %>%
             left_join(ae_meta[!duplicated(ae_meta$ID), ], by = "ID")  
     meta$sdtm2<-as.factor(as.character(sub(".; +","_", meta$sdtm)))
     #levels(meta$sdtm2)
      #convert to data frame
      data<-data_dtm2  %>% as.matrix() %>% 
          as.data.frame()  
      names(data)<-make.names(names(data))   
      train_feature<-names(data)
      saveRDS(train_feature, file="./train_data/ae_train_feature.RDS")

      
```




![png](output_10_1.png)


## Genearte training and validation sets


```R
set.seed(2018)

split <- createDataPartition(meta$sdtm, p=0.8, list=F)
train<- data[split,] 
test<- data[-split,] 

response_train <- meta[split,]$sdtm
response_test <- meta[-split,]$sdtm
response_train2 <- meta[split,]$sdtm2
response_test2 <- meta[-split,]$sdtm2

rm(ae_meta, data_dtm2, data,  split)
gc()
options(warn=0)
```


<table>
<thead><tr><th></th><th scope=col>used</th><th scope=col>(Mb)</th><th scope=col>gc trigger</th><th scope=col>(Mb)</th><th scope=col>max used</th><th scope=col>(Mb)</th></tr></thead>
<tbody>
	<tr><th scope=row>Ncells</th><td> 3094287</td><td>165.3   </td><td> 4578131</td><td>244.5   </td><td> 4578131</td><td>244.5   </td></tr>
	<tr><th scope=row>Vcells</th><td>35921542</td><td>274.1   </td><td>55485623</td><td>423.4   </td><td>44147749</td><td>336.9   </td></tr>
</tbody>
</table>



## Parameter tuning


```R
fitControl2 <- trainControl(## k-fold CV
                           method = "repeatedcv",
                           number = 10, #k
                           repeats = 5,
                           classProbs = TRUE, 
                           savePredictions = T,  summaryFunction = multiClassSummary)
#fitControl3 <- trainControl(## k-fold CV
#                           method = "cv",
#                           number = 10 #k
#                           )

```

## Class weight

The classes are unblanced therefore they are adjusted in random forest.


```R
casewt<-as.factor(response_train)
levels(casewt)<-c(max(summary(casewt))/summary(casewt))
```

## Train Models
Following models from `caret` will be trained. 
- SVM
A SVM is a discriminative classifier formally defined by a separating hyperplane. In other words, given labeled training data (supervised learning), the algorithm outputs an optimal hyperplane which categorizes new examples. In two dimentional space this hyperplane is a line dividing a plane in two parts where in each class lay in either side.

- logistic boosting
- CART
- Bagged Tree
- Random Forest
- Feed forward neuralnets
- XGBoost

## Multiclass SVM

Radial kernel $e(-\sigma|u-v|)^2$


```R
if(!file.exists("models/ae/svm.unigram.1.Rds")){
ptm<-proc.time()
set.seed(2018)

#svmGrid1
svm_mod1 <- train(x = train,
                       y = as.factor(response_train2),
                        method = "svmRadialSigma",
                          metric = "Accuracy",
                        trControl = fitControl2,
                          tuneLength = 9)
                        #tuneGrid=svmGrid1)
tms<-proc.time() - ptm


  
   plot_a<-plot(svm_mod1, main=paste("SVM - unigram -", round(tms[3]/60,1),"minutes"))
   plot_a
   print(svm_mod1)
   saveRDS(svm_mod1, "models/ae/svm.unigram.1.Rds")
   saveRDS(plot_a, "models/ae/plots/svm.plot.unigram.1.Rds")
} else{
  
      svm_mod1<-readRDS("models/ae/svm.unigram.1.Rds") 
    print(svm_mod1)
    print(readRDS("models/ae/plots/svm.plot.unigram.1.Rds"))

} 
```

    Support Vector Machines with Radial Basis Function Kernel 
    
    1398 samples
     432 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTP_AEENRF', 'AEENRTPT', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 5 times) 
    Summary of sample sizes: 1259, 1251, 1264, 1259, 1257, 1257, ... 
    Resampling results across tuning parameters:
    
      sigma         C   Accuracy   Kappa     
      2.980232e-08   1  0.4116654  0.00000000
      2.980232e-08   2  0.4116654  0.00000000
      2.980232e-08   4  0.4116654  0.00000000
      2.980232e-08   8  0.4116654  0.00000000
      2.980232e-08  16  0.4116654  0.00000000
      2.980232e-08  32  0.4116654  0.00000000
      9.536743e-07   1  0.4116654  0.00000000
      9.536743e-07   2  0.4116654  0.00000000
      9.536743e-07   4  0.4116654  0.00000000
      9.536743e-07   8  0.4116654  0.00000000
      9.536743e-07  16  0.4116654  0.00000000
      9.536743e-07  32  0.4116654  0.00000000
      3.051758e-05   1  0.4116654  0.00000000
      3.051758e-05   2  0.4116654  0.00000000
      3.051758e-05   4  0.4116654  0.00000000
      3.051758e-05   8  0.4116654  0.00000000
      3.051758e-05  16  0.4116654  0.00000000
      3.051758e-05  32  0.4287400  0.06366729
      9.765625e-04   1  0.4287400  0.06366729
      9.765625e-04   2  0.4287400  0.06366729
      9.765625e-04   4  0.4494513  0.13315362
      9.765625e-04   8  0.4646771  0.18291032
      9.765625e-04  16  0.5205110  0.34318842
      9.765625e-04  32  0.5847288  0.49798682
      3.125000e-02   1  0.5667282  0.45413606
      3.125000e-02   2  0.6121617  0.54240970
      3.125000e-02   4  0.6466011  0.60032549
      3.125000e-02   8  0.6625105  0.62748951
      3.125000e-02  16  0.6689626  0.63860273
      3.125000e-02  32  0.6702509  0.64040012
      1.000000e+00   1  0.6291440  0.56726066
      1.000000e+00   2  0.6340063  0.57635526
      1.000000e+00   4  0.6340063  0.57635526
      1.000000e+00   8  0.6340063  0.57635526
      1.000000e+00  16  0.6340063  0.57635526
      1.000000e+00  32  0.6340063  0.57635526
    
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were sigma = 0.03125 and C = 32.
    


![png](output_19_1.png)



```R
if(!file.exists("models/ae/svm.unigram.2.Rds")){
ptm<-proc.time()
set.seed(2018)
svmGrid1= expand.grid(sigma= seq(0.03,1,0.01), C= 2^c(5:10))
#svmGrid1
svm_mod2 <- train(x = train,
                       y = as.factor(response_train2),
                        method = "svmRadialSigma",
                          metric = "Accuracy",
                        trControl = fitControl2,
                        tuneGrid=svmGrid1)
tms<-proc.time() - ptm


  
   plot_a<-plot(svm_mod2, main=paste("SVM - unigram -", round(tms[3]/60,1),"minutes"))
   plot_a
   print(svm_mod2)
   saveRDS(svm_mod2, "models/ae/svm.unigram.2.Rds")
   saveRDS(plot_a, "models/ae/plots/svm.plot.unigram.2.Rds")
} else{
  
      svm_mod2<-readRDS("models/ae/svm.unigram.2.Rds") 
    print(svm_mod2)
    print(readRDS("models/ae/plots/svm.plot.unigram.2.Rds"))

} 
```

## LogitBoost

refer to sdtm_map5.Rmd and sdtm_map6.Rmd for how the hyper parameters were selected


```R
if(!file.exists("models/ae/lb.unigram.2.Rds")){
ptm<-proc.time()
set.seed(2018)
tuneGrid1<-data.frame(nIter=seq(15,23,1))
lb_mod2 <- train(x = train,
                       y = as.factor(response_train),
                        method = "LogitBoost",
                        trControl = fitControl2,
                        tuneGrid=tuneGrid1)
tms<-proc.time() - ptm


  
   plot_a<-plot(lb_mod2, main=paste("logistic boosting.2 - unigram -", round(tms[3]/60,1),"minutes"))
   plot_a
   print(lb_mod2) # nItr 21     0.9358211  0.8922967
   saveRDS(lb_mod2, "models/ae/lb.unigram.2.Rds")
   saveRDS(plot_a, "models/ae/lb.plot.unigram.2.Rds")
} else{
  
      lb_mod2<-readRDS("models/ae/lb.unigram.2.Rds") 
    print(lb_mod2)
    print(readRDS("models/ae/plots/lb.plot.unigram.2.Rds"))

} 
```

    Boosted Logistic Regression 
    
    1398 samples
     432 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTP_AEENRF', 'AEENRTPT', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 5 times) 
    Summary of sample sizes: 1259, 1251, 1264, 1259, 1257, 1257, ... 
    Resampling results across tuning parameters:
    
      nIter  Accuracy   Kappa    
      15     0.9234845  0.8653000
      16     0.9357899  0.8920164
      17     0.9304518  0.8796220
      18     0.9383524  0.8978558
      19     0.9326310  0.8856222
      20     0.9392563  0.9003152
      21     0.9359379  0.8924470
      22     0.9387655  0.9003992
      23     0.9357598  0.8934785
    
    Accuracy was used to select the optimal model using the largest value.
    The final value used for the model was nIter = 20.
    


![png](output_22_1.png)


## CART


```R
if(!file.exists("models/ae/cart.unigram.3.Rds")){
tuneGrid2<-data.frame(cp=seq(0,0.03,0.0001))
ptm<-proc.time()
set.seed(2018)
cart_mod3 <- train(x = train,
                        y = as.factor(response_train2),
                        method = "rpart",
                        trControl = fitControl2,
                        tuneGrid=tuneGrid2)
tms<-proc.time() - ptm
     plot_a<-plot(cart_mod3, main=paste("CART.3 - unigram -", round(tms[3]/60,1),"minutes"))
     plot_a
     #plot_b<-plot(cart_mod3$finalModel)+ text(cart_mod3$finalModel, cex=.7)
     #plot_c<-fancyRpartPlot(cart_mod3$finalModel, cex=.7)
     plot_d<-plot_ak(cart_mod3,"cp",w=0.0001, title=paste("CART.3 - unigram -", round(tms[3]/60,1),"minutes"))
     
     cp<-print(cart_mod3)
     plot_e<-plot(x=cp[,1],y=cp[,2],xlab = "cp",ylab="Accuracy (Repeated 10 fold CV)", main="CART.3 - unigram - Complexity Parameter Tuning")+
  lines(x=cp[,1],y=cp[,2],type="l")
     print(cart_mod3) # 
     saveRDS(cart_mod3, "models/ae/cart.unigram.3.Rds")
     saveRDS(plot_a, "models/ae/plots/cart.plot.a.unigram.3.Rds")
     saveRDS(plot_c, "models/ae/plots/cart.plot.c.unigram.3.Rds")
     saveRDS(plot_d, "models/ae/plots/cart.plot.d.unigram.3.Rds")
     saveRDS(plot_e, "models/ae/plots/cart.plot.e.unigram.3.Rds")
} else {     
    cart_mod3<-readRDS("models/ae/cart.unigram.3.Rds") 
  #  print(cart_mod3)
  #  print(readRDS("models/ae/plots/cart.plot.a.unigram.3.Rds"))
    print(readRDS("models/ae/plots/cart.plot.d.unigram.3.Rds")$plot ) 
  #  print(readRDS("models/ae/plots/cart.plot.e.unigram.3.Rds") ) 
}   
```


![png](output_24_0.png)


## Bagged Tree
Bootstrap aggregating, also called bagging, is a machine learning ensemble meta-algorithm designed to improve the stability and accuracy of machine learning algorithms used in statistical classification and regression. It also reduces variance and helps to avoid overfitting. Although it is usually applied to decision tree methods, it can be used with any type of method. Bagging is a special case of the model averaging approach.


```R
gc()
if(!file.exists("models/ae/bg.unigram.100.Rds")){
tree_bg<-list()
for (i in c(100)){
  ptm<-proc.time()
  set.seed(2018)
  bg_mod1 <- train(x = train, nbagg=i,
                          y = as.factor(response_train2),
                          method = "treebag",
                          trControl = fitControl2) #no tuning available
  tms<-proc.time() - ptm
     tree_bg[[paste("BG",toString(i))]]<-bg_mod1
     #plot_a<-plot(bg_mod1, main=paste("Tree Bag 1 - unigram - nbagg=",i, round(tms[3]/60,1),"minutes"))
     #plot_a
     print(bg_mod1) # 
     saveRDS(tree_bg, paste("models/ae/bg.tree.list.Rds"))
     saveRDS(bg_mod1, paste0("models/ae/bg.unigram.",i,".Rds"))
     }
} else {

    bg.100<-readRDS(paste0("models/ae/bg.unigram.",100,".Rds"))
    #bg.200<-readRDS(paste0("models/ae/bg.unigram.",200,".Rds"))
    print(bg.100)
    #print(bg.200)
}

```


<table>
<thead><tr><th></th><th scope=col>used</th><th scope=col>(Mb)</th><th scope=col>gc trigger</th><th scope=col>(Mb)</th><th scope=col>max used</th><th scope=col>(Mb)</th></tr></thead>
<tbody>
	<tr><th scope=row>Ncells</th><td>  3323564</td><td> 177.5   </td><td>  5985592</td><td> 319.7   </td><td>  5985592</td><td> 319.7   </td></tr>
	<tr><th scope=row>Vcells</th><td>508552684</td><td>3880.0   </td><td>708068114</td><td>5402.2   </td><td>510432267</td><td>3894.3   </td></tr>
</tbody>
</table>



    Bagged CART 
    
    1398 samples
     432 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTP_AEENRF', 'AEENRTPT', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 2 times) 
    Summary of sample sizes: 1259, 1251, 1264, 1259, 1257, 1257, ... 
    Resampling results:
    
      Accuracy   Kappa    
      0.8849649  0.8084448
    
    

## Random Forest


```R
gc()
```


<table>
<thead><tr><th></th><th scope=col>used</th><th scope=col>(Mb)</th><th scope=col>gc trigger</th><th scope=col>(Mb)</th><th scope=col>max used</th><th scope=col>(Mb)</th></tr></thead>
<tbody>
	<tr><th scope=row>Ncells</th><td>   6576019</td><td> 351.2    </td><td>  12225072</td><td> 652.9    </td><td>   6672412</td><td> 356.4    </td></tr>
	<tr><th scope=row>Vcells</th><td>1018989145</td><td>7774.3    </td><td>1229225385</td><td>9378.3    </td><td>1019061054</td><td>7774.9    </td></tr>
</tbody>
</table>




```R

if(!file.exists("models/ae/rf.unigram.1000.Rds")){
rf_list<-list()
for (i in c(500,1000,2000)){
#model 1
start<- proc.time()
set.seed(2018)
rf_mod1 <- train(x = train,
                 y = response_train2, method = "rf", ntree=i, 
                 metric="Accuracy", verbose=T,
                 trControl = trainControl(method="repeatedcv", number=10, repeats=5,classProbs = TRUE, 
                                          savePredictions = T,  summaryFunction = multiClassSummary), 
                 tuneGrid = expand.grid(.mtry = c(200,205,210)))

     
tms<-proc.time() - start
     plot_a<-plot(rf_mod1, main=paste("RF - unigram - tree=",i, round(tms[3]/60,1),"minutes"))
     plot_b<-plot_ak2(rf_mod1,"mtry",w=3, title=paste("RF - unigram - tree=",i, round(tms[3]/60,1),"minutes"))
     plot_c<-varImp(rf_mod1)
     #plot_d<-densityplot(rf_mod1)
     plot_a;plot_b
     print(rf_mod1) # 
     saveRDS(rf_mod1, paste0("models/ae/rf.unigram.",i,".Rds"))
     saveRDS(plot_a, paste0("models/ae/rf.plot.a.unigram.",i,".Rds"))
     saveRDS(plot_b, paste0("models/ae/rf.plot.b.unigram.",i,".Rds"))
     saveRDS(plot_c, paste0("models/ae/rf.varimp.c.unigram.",i,".Rds"))
     #saveRDS(plot_d, paste("models/ae/rf.density.d.unigram.",i,".Rds"))
     rf_list[[paste("rf.",toString(i))]]<-rf_mod1

} 
saveRDS(rf_list, paste("models/ae/rf_list.Rds"))
} else {
  
    #rf.100<-readRDS(paste("models/ae/bg.unigram.100.Rds")) 
    rf.500<-readRDS(paste("models/ae/rf.unigram.500.Rds")) 
    rf.1000<-readRDS(paste("models/ae/rf.unigram.1000.Rds")) 
    rf.2000<-readRDS(paste("models/ae/rf.unigram.2000.Rds")) 
          
    rf.500_default<- readRDS( paste("models/ae/plots/rf.plot.a.unigram.500.Rds"))
   # rf.500_mtry<- readRDS( paste("models/ae/plots/rf.plot.b.unigram.500.Rds"))
    #rf.500_varimp<- readRDS( paste("models/ae/plots/rf.varimp.c.unigram.500.Rds"))
    #rf.500_density<- saveRDS(plot_d, paste("models/ae/rf.density.d.unigram.500.Rds"))
    #rf.1000_default<- readRDS( paste("models/ae/plots/rf.plot.a.unigram.1000.Rds"))
    #rf.1000_mtry<- readRDS( paste("models/ae/plots/rf.plot.b.unigram.1000.Rds"))
    #rf.1000_varimp<- readRDS( paste("models/ae/plots/rf.varimp.c.unigram.1000.Rds"))
    #rf.1000_density<- saveRDS(plot_d, paste("models/ae/rf.density.d.unigram.1000.Rds"))


}


```


```R
rf.500
rf.500_default

```


    Random Forest 
    
    1398 samples
     432 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTP_AEENRF', 'AEENRTPT', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold, repeated 2 times) 
    Summary of sample sizes: 1111, 1123, 1117, 1117, 1124, 1113, ... 
    Resampling results across tuning parameters:
    
      mtry  logLoss    AUC        prAUC      Accuracy   Kappa      Mean_F1
      190   0.6987826  0.6941543  0.2339268  0.9116271  0.8577286  NaN    
      200   0.6995480  0.6941845  0.2364964  0.9116044  0.8576414  NaN    
      205   0.7248543  0.6938145  0.2356561  0.9143483  0.8619982  NaN    
      210   0.7263401  0.6938397  0.2354590  0.9125200  0.8588892  NaN    
      Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value
      NaN               0.9976284         NaN                  NaN                
      NaN               0.9976280         NaN                  NaN                
      NaN               0.9976693         NaN                  NaN                
      NaN               0.9976056         NaN                  NaN                
      Mean_Precision  Mean_Recall  Mean_Detection_Rate  Mean_Balanced_Accuracy
      NaN             NaN          0.01360638           NaN                   
      NaN             NaN          0.01360604           NaN                   
      NaN             NaN          0.01364699           NaN                   
      NaN             NaN          0.01361970           NaN                   
    
    Accuracy was used to select the optimal model using the largest value.
    The final value used for the model was mtry = 205.





![png](output_30_2.png)


## Weighted Random Forest


```R
if(!file.exists("models/ae/wt.rf.unigram.1000.Rds")){

for (i in c(500,1000)){
#model 1
start<- proc.time()
set.seed(2018)
rf_mod2 <- train(x = train,
                 y = response_train2, method = "rf", ntree=i, weights=casewt,
                 metric="Accuracy", verbose=T,
                 trControl = trainControl(method="repeatedcv", number=10, repeats=5,classProbs = TRUE, 
                                          savePredictions = T,  summaryFunction = multiClassSummary), 
                 tuneGrid = expand.grid(.mtry = c(200,205,210)))
tms<-proc.time() - start
     plot_a<-plot(rf_mod2, main=paste("Weighted RF - unigram - tree=",i, round(tms[3]/60,1),"minutes"))
     plot_b<-plot_ak2(rf_mod2,"mtry",w=3, title=paste("Weighted RF - unigram - tree=",i, round(tms[3]/60,1),"minutes"))
     plot_c<-varImp(rf_mod2)
     #plot_d<-densityplot(rf_mod2)
     plot_a;plot_b
     print(rf_mod2) # 
     saveRDS(rf_mod2, paste0("models/ae/wt.rf.unigram.",i,".Rds"))
     saveRDS(plot_a, paste0("models/ae/plots/wt.rf.plot.a.unigram.",i,".Rds"))
     saveRDS(plot_b, paste0("models/ae/plots/wt.rf.plot.b.unigram.",i,".Rds"))
     saveRDS(plot_c, paste0("models/ae/plots/wt.rf.varimp.c.unigram.",i,".Rds"))
     #saveRDS(plot_d, paste("models/ae/wt.rf.density.d.unigram.",i,".Rds"))
     rf_list[[paste("wt.rf.",toString(i))]]<-rf_mod2
}
    saveRDS(rf_list, paste("models/ae/rf_list.Rds"))
} else {
    wt.rf.500<-readRDS(paste("models/ae/wt.rf.unigram.500.Rds"))
    wt.rf.1000<-readRDS(paste("models/ae/wt.rf.unigram.1000.Rds")) 
    wt.rf.500_default<-  readRDS("models/ae/plots/wt.rf.plot.a.unigram.500.Rds")
    wt.rf.500_mtry<- readRDS("models/ae/plots/wt.rf.plot.b.unigram.500.Rds")
    wt.rf.500_varimp<-  readRDS("models/ae/plots/wt.rf.varimp.c.unigram.500.Rds")
    #rf.500_density<- saveRDS(plot_d, paste("models/ae/plots/wt.rf.density.d.unigram.5000.Rds"))
    wt.rf.1000_default<-  readRDS("models/ae/plots/wt.rf.plot.a.unigram.1000.Rds")
    #wt.rf.1000_mtry<-  readRDS("models/ae/plots/wt.rf.plot.b.unigram.1000.Rds")
    #wt.rf.1000_varimp<- readRDS("models/plots/ae/wt.rf.varimp.c.unigram.1000.Rds")
    #rf.1000_density<- saveRDS(plot_d, paste("models/ae/wt.rf.density.d.unigram.1000.Rds"))

}
wt.rf.500
wt.rf.500_default
```


    Random Forest 
    
    1398 samples
     433 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTP_AEENRF', 'AEENRTPT', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 5 times) 
    Summary of sample sizes: 1259, 1251, 1264, 1259, 1257, 1257, ... 
    Resampling results across tuning parameters:
    
      mtry  logLoss    AUC        prAUC      Accuracy   Kappa      Mean_F1
      200   0.6653039  0.8292926  0.1329652  0.9135437  0.8621886  NaN    
      205   0.6782674  0.8290754  0.1338157  0.9145790  0.8637015  NaN    
      210   0.6774311  0.8290553  0.1335245  0.9141721  0.8632385  NaN    
      Mean_Sensitivity  Mean_Specificity  Mean_Pos_Pred_Value  Mean_Neg_Pred_Value
      NaN               0.9977945         NaN                  NaN                
      NaN               0.9977978         NaN                  NaN                
      NaN               0.9978141         NaN                  NaN                
      Mean_Precision  Mean_Recall  Mean_Detection_Rate  Mean_Balanced_Accuracy
      NaN             NaN          0.01363498           NaN                   
      NaN             NaN          0.01365043           NaN                   
      NaN             NaN          0.01364436           NaN                   
    
    Accuracy was used to select the optimal model using the largest value.
    The final value used for the model was mtry = 205.





![png](output_32_2.png)


## Feed forward neural network


```R
if(!file.exists("models/ae/nnet_mod3.Rds")){

#ptm<-proc.time()
#set.seed(2018)
#train_n <-predict(preProcess(x = train,method="BoxCox"), train)
#set.seed(2018)
#nnet_mod1 <- train(x = train,
#                    y = as.factor(response_train),
#                    method = "nnet",
#                    trControl = fitControl2,tuneLength=3)
#nn_tm1<-proc.time() - ptm
#
#print(nnet_mod1) 
#
##size=10
#ptm<-proc.time()
#set.seed(2018)
#nnet_mod <- train(x = train,
#                    y = as.factor(response_train),
#                    method = "nnet",
#                    trControl = fitControl2,
#                  tuneGrid = expand.grid(size = 10,
#                   decay = 0.1),
#                    MaxNWts = 100100)
#nn_tm<-proc.time() - ptm
#
#print(nnet_mod)
#
    
ptm<-proc.time()    
set.seed(2018)
nnet_mod3 <- train(x = train,
                    y = as.factor(response_train),
                    method = "nnet",
                    trControl = fitControl2,
                  tuneGrid = expand.grid(size = 50,
                   decay = 0.1),
                    MaxNWts = 100100)
nn_tm3<-proc.time() - ptm

print(nnet_mod3)
  


saveRDS(nnet_mod3, "models/ae/nnet_mod3.Rds")
nnet_mod<-readRDS("models/ae/nnet_mod3.Rds")
} else {

nnet_mod<-readRDS("models/ae/nnet_mod3.Rds")
print(nnet_mod)
}    
```

    Neural Network 
    
    1398 samples
     432 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTPT', 'AEENRTPT; AEENRF', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 5 times) 
    Summary of sample sizes: 1260, 1251, 1264, 1259, 1257, 1258, ... 
    Resampling results:
    
      Accuracy   Kappa    
      0.6782632  0.6544387
    
    Tuning parameter 'size' was held constant at a value of 50
    Tuning
     parameter 'decay' was held constant at a value of 0.1
    

## Gradient Boosting via XGBoost
[XGBoost](https://github.com/dmlc/xgboost) is a gradient boosting framework that is "Scalable, Portable and Distributed Gradient Boosting (GBM, GBRT, GBDT) Library"



```R
 if(!file.exists("models/ae/xgb.1.RDS")){
ptm<-proc.time()
set.seed(2018)
xgb_m1_1 <- train(x = train,
                  y = response_train2,
                    method = "xgbTree", 
                      booster="gbtree",
                      tree_method=c("exact"),metric = "Accuracy", 
                      tuneGrid=expand.grid(
                      nrounds=1000,
                      max_depth=c(6,7,8),
                      eta = 0.001, #learning rate
                      gamma=c(1,2),
                      colsample_bytree=c(0.75),
                      min_child_weight=c(1,2),
                      subsample=c(.75)
                            ), 

                  trControl = trainControl(method="repeatedcv", number=10, repeats=5,
                                           classProbs = T, 
                                           savePredictions = T,  
                                           summaryFunction = multiClassSummary)
                 )
tms <-proc.time() - ptm   ;
saveRDS(xgb_m1_1, file="models/ae/xgb.1.RDS")
xgb_m1_1<- readRDS("models/ae/xgb.1.RDS")
min<-tms[3]/60 
title<-paste("XGBoost - elapsed", min, " min")
plot.a<-plot(xgb_m1_1,main=title)
saveRDS(plot.a, file="models/ae/plots/xgb.1.plot.RDS")
print(xgb_m1_1)
} else {
xgb_m1_1<- readRDS("models/ae/xgb.1.RDS")
xgb_plot<- readRDS("models/ae/plots/xgb.1.plot.RDS")
print(xgb_m1_1)
xgb_plot
}
```

    eXtreme Gradient Boosting 
    
    1398 samples
     432 predictor
      67 classes: 'AEACN', 'AEACNDEV', 'AEACNOTH', 'AECAT', 'AECONTRT', 'AEDECOD', 'AEDIR', 'AEDUR', 'AEENDTC', 'AEENDTC_TM', 'AEENDY', 'AEENRF', 'AEENRTP_AEENRF', 'AEENRTPT', 'AEENTPT', 'AEHLGT', 'AEHLGTCD', 'AEHLT', 'AEHLTCD', 'AELAT', 'AELIFTH', 'AELLT', 'AELLTCD', 'AELOC', 'AEMODIFY', 'AEOUT', 'AEPATT', 'AEPORTOT', 'AEPRESP', 'AEPTCD', 'AEREL', 'AERELNST', 'AESCAN', 'AESCAT', 'AESCONG', 'AESDISAB', 'AESDTH', 'AESER', 'AESEV', 'AESHOSP', 'AESLIFE', 'AESMIE', 'AESOC', 'AESOCCD', 'AESOD', 'AESPID', 'AESTDTC', 'AESTDTC_TM', 'AESTRF', 'AETERM', 'AETOXGR', 'DROP', 'DTHDTC', 'FAORRES', 'QNAM_AESI', 'QNAM_COPD', 'QNAM_ESAM1', 'QNAM_ESAM2', 'QNAM_ESAM3', 'QNAM_EXER', 'QNAM_EXLAB', 'QNAM_EXSAB', 'QNAM_EXSTER', 'SITEID', 'STUDYID', 'SUBJID', 'SUPPAE.QVAL' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 5 times) 
    Summary of sample sizes: 1259, 1251, 1264, 1259, 1257, 1257, ... 
    Resampling results across tuning parameters:
    
      max_depth  gamma  min_child_weight  nrounds  logLoss   AUC        prAUC     
      6          1      1                 1000     2.065130  0.8983989  0.09907956
      6          1      1                 2000     1.629374  0.9118170  0.10277275
      6          1      2                 1000     2.209955  0.8742075  0.09502308
      6          1      2                 2000     1.785216  0.8999165  0.09598909
      6          2      1                 1000     2.075581  0.8970426  0.09773316
      6          2      1                 2000     1.648920  0.9083801  0.10121332
      6          2      2                 1000     2.220563  0.8738457  0.09384574
      6          2      2                 2000     1.801472  0.8978097  0.09601744
      7          1      1                 1000     2.055922  0.9003584  0.10002706
      7          1      1                 2000     1.616792  0.9140793  0.10352352
      7          1      2                 1000     2.202257  0.8765249  0.09518742
      7          1      2                 2000     1.773964  0.9015094  0.09680103
      7          2      1                 1000     2.066791  0.8994228  0.09867158
      7          2      1                 2000     1.636570  0.9114973  0.10148885
      7          2      2                 1000     2.213113  0.8755663  0.09436044
      7          2      2                 2000     1.790054  0.8996095  0.09687580
      8          1      1                 1000     2.048090  0.9009491  0.10016716
      8          1      1                 2000     1.606008  0.9149787  0.10383312
      8          1      2                 1000     2.195701  0.8777804  0.09615717
      8          1      2                 2000     1.764161  0.9022110  0.09719059
      8          2      1                 1000     2.058999  0.8999283  0.09969102
      8          2      1                 2000     1.626108  0.9124091  0.10251175
      8          2      2                 1000     2.206467  0.8771982  0.09478339
      8          2      2                 2000     1.780095  0.8999736  0.09760230
      Accuracy   Kappa      Mean_F1  Mean_Sensitivity  Mean_Specificity
      0.6525839  0.2833520  NaN      NaN               0.9886952       
      0.6956529  0.3978389  NaN      NaN               0.9900944       
      0.6016935  0.1272335  NaN      NaN               0.9868748       
      0.6505236  0.2781942  NaN      NaN               0.9886438       
      0.6537407  0.2843305  NaN      NaN               0.9886757       
      0.6927772  0.3889268  NaN      NaN               0.9899582       
      0.6014037  0.1266514  NaN      NaN               0.9868701       
      0.6519759  0.2806983  NaN      NaN               0.9886520       
      0.6527268  0.2849662  NaN      NaN               0.9887300       
      0.6962164  0.4007626  NaN      NaN               0.9901574       
      0.6023949  0.1307190  NaN      NaN               0.9869218       
      0.6519789  0.2827945  NaN      NaN               0.9887038       
      0.6528794  0.2840285  NaN      NaN               0.9886983       
      0.6945154  0.3954044  NaN      NaN               0.9900743       
      0.6024029  0.1301617  NaN      NaN               0.9869118       
      0.6519862  0.2821053  NaN      NaN               0.9886842       
      0.6541528  0.2900925  NaN      NaN               0.9888061       
      0.6977881  0.4062170  NaN      NaN               0.9902536       
      0.6031189  0.1333027  NaN      NaN               0.9869511       
      0.6538226  0.2892377  NaN      NaN               0.9887976       
      0.6537273  0.2876836  NaN      NaN               0.9887603       
      0.6953649  0.3999925  NaN      NaN               0.9901786       
      0.6038336  0.1347467  NaN      NaN               0.9869592       
      0.6522784  0.2832388  NaN      NaN               0.9887026       
      Mean_Pos_Pred_Value  Mean_Neg_Pred_Value  Mean_Precision  Mean_Recall
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      NaN                  NaN                  NaN             NaN        
      Mean_Detection_Rate  Mean_Balanced_Accuracy
      0.009740058          NaN                   
      0.010382879          NaN                   
      0.008980500          NaN                   
      0.009709307          NaN                   
      0.009757323          NaN                   
      0.010339958          NaN                   
      0.008976175          NaN                   
      0.009730984          NaN                   
      0.009742192          NaN                   
      0.010391289          NaN                   
      0.008990969          NaN                   
      0.009731029          NaN                   
      0.009744469          NaN                   
      0.010365902          NaN                   
      0.008991088          NaN                   
      0.009731138          NaN                   
      0.009763474          NaN                   
      0.010414748          NaN                   
      0.009001775          NaN                   
      0.009758546          NaN                   
      0.009757124          NaN                   
      0.010378580          NaN                   
      0.009012441          NaN                   
      0.009735498          NaN                   
    
    Tuning parameter 'eta' was held constant at a value of 0.001
    Tuning
     parameter 'colsample_bytree' was held constant at a value of 0.75
    
    Tuning parameter 'subsample' was held constant at a value of 0.75
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were nrounds = 2000, max_depth = 8, eta
     = 0.001, gamma = 1, colsample_bytree = 0.75, min_child_weight = 1
     and subsample = 0.75.
    




![png](output_36_2.png)


# Dotplot


```R
models<-list()
models[["SVM"]]<-svm_mod1
models[["FFNN"]]<-nnet_mod
models[["XGB"]]<-xgb_m1_1


#models[["LogitBoost"]]<-lb_mod2
#models[["CART"]]<-cart_mod3
#models[["BaggedCART"]]<-bg.100
#models[["RF"]]<-rf.500
#models[["Wt.RF"]]<-wt.rf.500
```


```R
a<- resamples(models)
dotplot(a)
```

    Warning message in resamples.default(models):
    "Some performance measures were not computed for each model: AUC, logLoss, Mean_Balanced_Accuracy, Mean_Detection_Rate, Mean_F1, Mean_Neg_Pred_Value, Mean_Pos_Pred_Value, Mean_Precision, Mean_Recall, Mean_Sensitivity, Mean_Specificity, prAUC"




![png](output_39_2.png)


## Test

**logistic boosing**


```R

prediction <-predict(lb_mod2, newdata=test)
cm.lb_mod2 <-confusionMatrix(prediction, as.factor(response_test))
cm.lb_mod2
#ft<-nnet_p== response_test
#result<-data.frame(meta[-split,]$ID, response_test,nnet_p,ft)
```

    Warning message in confusionMatrix.default(prediction, as.factor(response_test)):
    "The data contains levels not found in the data, but they are empty and will be dropped."Warning message in levels(reference) != levels(data):
    "longer object length is not a multiple of shorter object length"Warning message in confusionMatrix.default(prediction, as.factor(response_test)):
    "Levels are not in the same order for reference and data. Refactoring data to match."


    Confusion Matrix and Statistics
    
                      Reference
    Prediction         AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN                3        0        0     0        0       0     0     0
      AEACNDEV             0        0        0     0        0       0     0     0
      AEACNOTH             0        0        0     0        0       0     0     0
      AECAT                0        0        0     0        0       0     0     0
      AECONTRT             0        0        0     0        2       0     0     0
      AEDECOD              0        0        0     0        0       3     0     0
      AEDIR                0        0        0     0        0       0     0     0
      AEDUR                0        0        0     0        0       0     0     0
      AEENDTC              0        0        0     0        0       0     0     0
      AEENDTC_TM           0        0        0     0        0       0     0     0
      AEENDY               0        0        0     0        0       0     0     0
      AEENRF               0        0        0     0        0       0     0     0
      AEENRTPT             0        0        0     0        0       0     0     0
      AEENRTPT; AEENRF     0        0        0     0        0       0     0     0
      AEENTPT              0        0        0     0        0       0     0     0
      AEHLGT               0        0        0     0        0       0     0     0
      AEHLGTCD             0        0        0     0        0       0     0     0
      AEHLT                0        0        0     0        0       0     0     0
      AEHLTCD              0        0        0     0        0       0     0     0
      AELAT                0        0        0     0        0       0     0     0
      AELIFTH              0        0        0     0        0       0     0     0
      AELLT                0        0        0     0        0       0     0     0
      AELLTCD              0        0        0     0        0       0     0     0
      AELOC                0        0        0     0        0       0     0     0
      AEMODIFY             0        0        0     0        0       0     0     0
      AEOUT                0        0        0     0        0       0     0     0
      AEPATT               0        0        0     0        0       0     0     0
      AEPORTOT             0        0        0     0        0       0     0     0
      AEPRESP              0        0        0     0        0       0     0     0
      AEPTCD               0        0        0     0        0       0     0     0
      AEREL                0        0        0     0        0       0     0     0
      AERELNST             0        0        0     0        0       0     0     0
      AESCAN               0        0        0     0        0       0     0     0
      AESCAT               0        0        0     0        0       0     0     0
      AESCONG              0        0        0     0        0       0     0     0
      AESDISAB             0        0        0     0        0       0     0     0
      AESDTH               0        0        0     0        0       0     0     0
      AESER                0        0        0     0        0       0     0     0
      AESEV                0        0        0     0        0       0     0     0
      AESHOSP              0        0        0     0        0       0     0     0
      AESLIFE              0        0        0     0        0       0     0     0
      AESMIE               0        0        0     0        0       0     0     0
      AESOC                0        0        0     0        0       0     0     0
      AESOCCD              0        0        0     0        0       0     0     0
      AESOD                0        0        0     0        0       0     0     0
      AESPID               0        0        0     0        0       0     0     0
      AESTDTC              0        0        0     0        0       0     0     0
      AESTDTC_TM           0        0        0     0        0       0     0     0
      AESTRF               0        0        0     0        0       0     0     0
      AETERM               0        0        0     0        0       0     0     0
      AETOXGR              0        0        0     0        0       0     0     0
      DROP                 0        0        0     0        0       0     0     0
      DTHDTC               0        0        0     0        0       0     0     0
      FAORRES              0        0        0     0        0       0     0     0
      QNAM_AESI            0        0        0     0        0       0     0     0
      QNAM_COPD            0        0        0     0        0       0     0     0
      QNAM_ESAM1           0        0        0     0        0       0     0     0
      QNAM_ESAM2           0        0        0     0        0       0     0     0
      QNAM_ESAM3           0        0        0     0        0       0     0     0
      QNAM_EXER            0        0        0     0        0       0     0     0
      QNAM_EXLAB           0        0        0     0        0       0     0     0
      QNAM_EXSAB           0        0        0     0        0       0     0     0
      QNAM_EXSTER          0        0        0     0        0       0     0     0
      SITEID               0        0        0     0        0       0     0     0
      STUDYID              1        0        0     0        0       0     0     0
      SUBJID               0        0        0     0        0       0     0     0
      SUPPAE.QVAL          0        0        0     0        0       0     0     0
                      Reference
    Prediction         AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTPT AEENRTPT; AEENRF
      AEACN                  0          0      0      0        0                0
      AEACNDEV               0          0      0      0        0                0
      AEACNOTH               0          0      0      0        0                0
      AECAT                  0          0      0      0        0                0
      AECONTRT               0          0      0      0        0                0
      AEDECOD                0          0      0      0        0                0
      AEDIR                  0          0      0      0        0                0
      AEDUR                  0          0      0      0        0                0
      AEENDTC                2          0      0      0        0                0
      AEENDTC_TM             0          1      0      0        0                0
      AEENDY                 0          0      0      0        0                0
      AEENRF                 0          0      0      0        0                0
      AEENRTPT               0          0      0      0        4                0
      AEENRTPT; AEENRF       0          0      0      0        0                0
      AEENTPT                0          0      0      0        0                0
      AEHLGT                 0          0      0      0        0                0
      AEHLGTCD               0          0      0      0        0                0
      AEHLT                  0          0      0      0        0                0
      AEHLTCD                0          0      0      0        0                0
      AELAT                  0          0      0      0        0                0
      AELIFTH                0          0      0      0        0                0
      AELLT                  0          0      0      0        0                0
      AELLTCD                0          0      0      0        0                0
      AELOC                  0          0      0      0        0                0
      AEMODIFY               0          0      0      0        0                0
      AEOUT                  0          0      0      0        0                0
      AEPATT                 0          0      0      0        0                0
      AEPORTOT               0          0      0      0        0                0
      AEPRESP                0          0      0      0        0                0
      AEPTCD                 0          0      0      0        0                0
      AEREL                  0          0      0      0        0                0
      AERELNST               0          0      0      0        0                0
      AESCAN                 0          0      0      0        0                0
      AESCAT                 0          0      0      0        0                0
      AESCONG                0          0      0      0        0                0
      AESDISAB               0          0      0      0        0                0
      AESDTH                 0          0      0      0        0                0
      AESER                  0          0      0      0        0                0
      AESEV                  0          0      0      0        0                0
      AESHOSP                0          0      0      0        0                0
      AESLIFE                0          0      0      0        0                0
      AESMIE                 0          0      0      0        0                0
      AESOC                  0          0      0      0        0                0
      AESOCCD                0          0      0      0        0                0
      AESOD                  0          0      0      0        0                0
      AESPID                 0          0      0      0        0                0
      AESTDTC                0          0      0      0        0                0
      AESTDTC_TM             0          0      0      0        0                0
      AESTRF                 0          0      0      0        0                0
      AETERM                 0          0      0      0        0                0
      AETOXGR                0          0      0      0        0                0
      DROP                   2          0      0      0        0                0
      DTHDTC                 0          0      0      0        0                0
      FAORRES                0          0      0      0        0                0
      QNAM_AESI              0          0      0      0        0                0
      QNAM_COPD              0          0      0      0        0                0
      QNAM_ESAM1             0          0      0      0        0                0
      QNAM_ESAM2             0          0      0      0        0                0
      QNAM_ESAM3             0          0      0      0        0                0
      QNAM_EXER              0          0      0      0        0                0
      QNAM_EXLAB             0          0      0      0        0                0
      QNAM_EXSAB             0          0      0      0        0                0
      QNAM_EXSTER            0          0      0      0        0                0
      SITEID                 0          0      0      0        0                0
      STUDYID                0          0      0      0        0                0
      SUBJID                 0          0      0      0        0                0
      SUPPAE.QVAL            0          0      0      0        0                0
                      Reference
    Prediction         AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                  0      0        0     0       0     0       0     0
      AEACNDEV               0      0        0     0       0     0       0     0
      AEACNOTH               0      0        0     0       0     0       0     0
      AECAT                  0      0        0     0       0     0       0     0
      AECONTRT               0      0        0     0       0     0       0     0
      AEDECOD                0      0        0     0       0     0       0     0
      AEDIR                  0      0        0     0       0     0       0     0
      AEDUR                  0      0        0     0       0     0       0     0
      AEENDTC                0      0        0     0       0     0       0     0
      AEENDTC_TM             0      0        0     0       0     0       0     0
      AEENDY                 0      0        0     0       0     0       0     0
      AEENRF                 0      0        0     0       0     0       0     0
      AEENRTPT               0      0        0     0       0     0       0     0
      AEENRTPT; AEENRF       0      0        0     0       0     0       0     0
      AEENTPT                0      0        0     0       0     0       0     0
      AEHLGT                 0      4        0     1       0     0       0     0
      AEHLGTCD               0      0        3     0       0     0       0     0
      AEHLT                  0      0        0     3       0     0       0     0
      AEHLTCD                0      0        1     0       4     0       0     0
      AELAT                  0      0        0     0       0     0       0     0
      AELIFTH                0      0        0     0       0     0       2     0
      AELLT                  0      0        0     0       0     0       0     4
      AELLTCD                0      0        0     0       0     0       0     0
      AELOC                  0      0        0     0       0     0       0     0
      AEMODIFY               0      0        0     0       0     0       0     0
      AEOUT                  0      0        0     0       0     0       0     0
      AEPATT                 0      0        0     0       0     0       0     0
      AEPORTOT               0      0        0     0       0     0       0     0
      AEPRESP                0      0        0     0       0     0       0     0
      AEPTCD                 0      0        0     0       0     0       0     0
      AEREL                  0      0        0     0       0     0       0     0
      AERELNST               0      0        0     0       0     0       0     0
      AESCAN                 0      0        0     0       0     0       0     0
      AESCAT                 0      0        0     0       0     0       0     0
      AESCONG                0      0        0     0       0     0       0     0
      AESDISAB               0      0        0     0       0     0       0     0
      AESDTH                 0      0        0     0       0     0       0     0
      AESER                  0      0        0     0       0     0       0     0
      AESEV                  0      0        0     0       0     0       0     0
      AESHOSP                0      0        0     0       0     0       0     0
      AESLIFE                0      0        0     0       0     0       0     0
      AESMIE                 0      0        0     0       0     0       0     0
      AESOC                  0      0        0     0       0     0       0     0
      AESOCCD                0      0        0     0       0     0       0     0
      AESOD                  0      0        0     0       0     0       0     0
      AESPID                 0      0        0     0       0     0       0     0
      AESTDTC                0      0        0     0       0     0       0     0
      AESTDTC_TM             0      0        0     0       0     0       0     0
      AESTRF                 0      0        0     0       0     0       0     0
      AETERM                 0      0        0     0       0     0       0     0
      AETOXGR                0      0        0     0       0     0       0     0
      DROP                   0      0        0     0       0     0       0     0
      DTHDTC                 0      0        0     0       0     0       0     0
      FAORRES                0      0        0     0       0     0       0     0
      QNAM_AESI              0      0        0     0       0     0       0     0
      QNAM_COPD              0      0        0     0       0     0       0     0
      QNAM_ESAM1             0      0        0     0       0     0       0     0
      QNAM_ESAM2             0      0        0     0       0     0       0     0
      QNAM_ESAM3             0      0        0     0       0     0       0     0
      QNAM_EXER              0      0        0     0       0     0       0     0
      QNAM_EXLAB             0      0        0     0       0     0       0     0
      QNAM_EXSAB             0      0        0     0       0     0       0     0
      QNAM_EXSTER            0      0        0     0       0     0       0     0
      SITEID                 0      0        0     0       0     0       0     0
      STUDYID                0      0        0     0       0     0       0     0
      SUBJID                 0      0        0     0       0     0       0     0
      SUPPAE.QVAL            0      0        0     0       0     0       0     0
                      Reference
    Prediction         AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                  0     0        0     0      0        0       0      0
      AEACNDEV               0     0        0     0      0        0       0      0
      AEACNOTH               0     0        0     0      0        0       0      0
      AECAT                  0     0        0     0      0        0       0      0
      AECONTRT               0     0        0     0      0        0       0      0
      AEDECOD                0     0        0     0      0        0       0      0
      AEDIR                  0     0        0     0      0        0       0      0
      AEDUR                  0     0        0     0      0        0       0      0
      AEENDTC                0     0        0     0      0        0       0      0
      AEENDTC_TM             0     0        0     0      0        0       0      0
      AEENDY                 0     0        0     0      0        0       0      0
      AEENRF                 0     0        0     0      0        0       0      0
      AEENRTPT               0     0        0     0      0        0       0      0
      AEENRTPT; AEENRF       0     0        0     0      0        0       0      0
      AEENTPT                0     0        0     0      0        0       0      0
      AEHLGT                 0     0        0     0      0        0       0      0
      AEHLGTCD               0     0        0     0      0        0       0      0
      AEHLT                  0     0        0     0      0        0       0      0
      AEHLTCD                0     0        0     0      0        0       0      1
      AELAT                  0     0        0     0      0        0       0      0
      AELIFTH                0     0        0     0      0        0       0      0
      AELLT                  0     0        0     0      0        0       0      0
      AELLTCD                1     0        0     0      0        0       0      0
      AELOC                  0     0        0     0      0        0       0      0
      AEMODIFY               0     0        0     0      0        0       0      0
      AEOUT                  0     0        0     4      0        0       0      0
      AEPATT                 0     0        0     0      3        0       0      0
      AEPORTOT               0     0        0     0      0        0       0      0
      AEPRESP                0     0        0     0      0        0       0      0
      AEPTCD                 0     0        0     0      0        0       0      3
      AEREL                  0     0        0     0      0        0       0      0
      AERELNST               0     0        0     0      0        0       0      0
      AESCAN                 0     0        0     0      0        0       0      0
      AESCAT                 0     0        0     0      0        0       0      0
      AESCONG                0     0        0     0      0        0       0      0
      AESDISAB               0     0        0     0      0        0       0      0
      AESDTH                 0     0        0     0      0        0       0      0
      AESER                  0     0        0     0      0        0       0      0
      AESEV                  0     0        0     0      0        0       0      0
      AESHOSP                0     0        0     0      0        0       0      0
      AESLIFE                0     0        0     0      0        0       0      0
      AESMIE                 0     0        0     0      0        0       0      0
      AESOC                  0     0        0     0      0        0       0      0
      AESOCCD                0     0        0     0      0        0       0      0
      AESOD                  0     0        0     0      0        0       0      0
      AESPID                 0     0        0     0      0        0       0      0
      AESTDTC                0     0        0     0      0        0       0      0
      AESTDTC_TM             0     0        0     0      0        0       0      0
      AESTRF                 0     0        0     0      0        0       0      0
      AETERM                 0     0        0     0      0        0       0      0
      AETOXGR                0     0        0     0      0        0       0      0
      DROP                   0     0        0     0      1        0       0      0
      DTHDTC                 0     0        0     0      0        0       0      0
      FAORRES                0     0        0     0      0        0       0      0
      QNAM_AESI              0     0        0     0      0        0       0      0
      QNAM_COPD              0     0        0     0      0        0       0      0
      QNAM_ESAM1             0     0        0     0      0        0       0      0
      QNAM_ESAM2             0     0        0     0      0        0       0      0
      QNAM_ESAM3             0     0        0     0      0        0       0      0
      QNAM_EXER              0     0        0     0      0        0       0      0
      QNAM_EXLAB             0     0        0     0      0        0       0      0
      QNAM_EXSAB             0     0        0     0      0        0       0      0
      QNAM_EXSTER            0     0        0     0      0        0       0      0
      SITEID                 0     0        0     0      0        0       0      0
      STUDYID                0     0        0     0      0        0       0      0
      SUBJID                 0     0        0     0      0        0       0      0
      SUPPAE.QVAL            0     0        0     0      0        0       0      0
                      Reference
    Prediction         AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN                0        0      0      0       0        0      0     0
      AEACNDEV             0        0      0      0       0        0      0     0
      AEACNOTH             0        0      0      0       0        0      0     0
      AECAT                0        0      0      0       0        0      0     0
      AECONTRT             0        0      0      0       0        0      0     0
      AEDECOD              0        0      0      0       0        0      0     0
      AEDIR                0        0      0      0       0        0      0     0
      AEDUR                0        0      0      0       0        0      0     0
      AEENDTC              0        0      0      0       0        0      0     0
      AEENDTC_TM           0        0      0      0       0        0      0     0
      AEENDY               0        0      0      0       0        0      0     0
      AEENRF               0        0      0      0       0        0      0     0
      AEENRTPT             0        0      0      0       0        0      0     0
      AEENRTPT; AEENRF     0        0      0      0       0        0      0     0
      AEENTPT              0        0      0      0       0        0      0     0
      AEHLGT               0        0      0      0       0        0      0     0
      AEHLGTCD             0        0      0      0       0        0      0     0
      AEHLT                0        0      0      0       0        0      0     0
      AEHLTCD              0        0      0      0       0        0      0     0
      AELAT                0        0      0      0       0        0      0     0
      AELIFTH              0        0      0      0       0        0      0     0
      AELLT                0        0      0      0       0        0      0     0
      AELLTCD              0        0      0      0       0        0      0     0
      AELOC                0        0      0      0       0        0      0     0
      AEMODIFY             0        0      0      0       0        0      0     0
      AEOUT                0        0      0      0       0        0      0     0
      AEPATT               0        0      0      0       0        0      0     0
      AEPORTOT             0        0      0      0       0        0      0     0
      AEPRESP              0        0      0      0       0        0      0     0
      AEPTCD               0        0      0      0       0        0      0     0
      AEREL                4        0      0      0       0        0      0     0
      AERELNST             0        0      0      0       0        0      0     0
      AESCAN               0        0      0      0       0        0      0     0
      AESCAT               0        0      0      0       0        0      0     0
      AESCONG              0        0      0      0       2        0      0     0
      AESDISAB             0        0      0      0       0        3      0     0
      AESDTH               0        0      0      0       0        0      0     0
      AESER                0        0      0      0       0        0      0     3
      AESEV                0        0      0      0       0        0      0     0
      AESHOSP              0        0      0      0       0        0      0     0
      AESLIFE              0        0      0      0       0        0      0     0
      AESMIE               0        0      0      0       0        0      0     0
      AESOC                0        0      0      0       0        0      0     0
      AESOCCD              0        0      0      0       0        0      0     0
      AESOD                0        0      0      0       0        0      0     0
      AESPID               0        0      0      0       0        0      0     0
      AESTDTC              0        0      0      0       0        0      0     0
      AESTDTC_TM           0        0      0      0       0        0      0     0
      AESTRF               0        0      0      0       0        0      0     0
      AETERM               0        0      0      0       0        0      0     0
      AETOXGR              0        0      0      0       0        0      0     0
      DROP                 0        0      0      0       0        0      1     1
      DTHDTC               0        0      0      0       0        0      0     0
      FAORRES              0        0      0      0       0        0      0     0
      QNAM_AESI            0        0      0      0       0        0      0     0
      QNAM_COPD            0        0      0      0       0        0      0     0
      QNAM_ESAM1           0        0      0      0       0        0      0     0
      QNAM_ESAM2           0        0      0      0       0        0      0     0
      QNAM_ESAM3           0        0      0      0       0        0      0     0
      QNAM_EXER            0        0      0      0       0        0      0     0
      QNAM_EXLAB           0        0      0      0       0        0      0     0
      QNAM_EXSAB           0        0      0      0       0        0      0     0
      QNAM_EXSTER          0        0      0      0       0        0      0     0
      SITEID               0        0      0      0       0        0      0     0
      STUDYID              0        0      0      0       0        0      0     0
      SUBJID               0        0      0      0       0        0      0     0
      SUPPAE.QVAL          0        0      0      0       0        0      0     0
                      Reference
    Prediction         AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN                0       0       0      0     0       0     0      0
      AEACNDEV             0       0       0      0     0       0     0      0
      AEACNOTH             0       0       0      0     0       0     0      0
      AECAT                0       0       0      0     0       0     0      0
      AECONTRT             0       0       0      0     0       0     0      0
      AEDECOD              0       0       0      0     0       0     0      0
      AEDIR                0       0       0      0     0       0     0      0
      AEDUR                0       0       0      0     0       0     0      0
      AEENDTC              0       0       0      0     0       0     0      0
      AEENDTC_TM           0       0       0      0     0       0     0      0
      AEENDY               0       0       0      0     0       0     0      0
      AEENRF               0       0       0      0     0       0     0      0
      AEENRTPT             0       0       0      0     0       0     0      0
      AEENRTPT; AEENRF     0       0       0      0     0       0     0      0
      AEENTPT              0       0       0      0     0       0     0      0
      AEHLGT               0       0       0      0     0       0     0      0
      AEHLGTCD             0       0       0      0     0       0     0      0
      AEHLT                0       0       0      0     0       0     0      0
      AEHLTCD              0       0       0      0     0       0     0      0
      AELAT                0       0       0      0     0       0     0      0
      AELIFTH              0       0       0      0     0       0     0      0
      AELLT                0       0       0      0     0       0     0      0
      AELLTCD              0       0       0      0     0       0     0      0
      AELOC                0       0       0      0     0       0     0      0
      AEMODIFY             0       0       0      0     0       0     0      0
      AEOUT                0       0       0      0     0       0     0      0
      AEPATT               0       0       0      0     0       0     0      0
      AEPORTOT             0       0       0      0     0       0     0      0
      AEPRESP              0       0       0      0     0       0     0      0
      AEPTCD               0       0       0      0     0       0     0      0
      AEREL                0       0       0      0     0       0     0      0
      AERELNST             0       0       0      0     0       0     0      0
      AESCAN               0       0       0      0     0       0     0      0
      AESCAT               0       0       0      0     0       0     0      0
      AESCONG              0       0       0      0     0       0     0      0
      AESDISAB             0       0       0      0     0       0     0      0
      AESDTH               0       0       0      0     0       0     0      0
      AESER                0       0       0      0     0       0     0      0
      AESEV                4       0       0      0     0       0     0      0
      AESHOSP              0       3       0      0     0       0     0      0
      AESLIFE              0       0       0      0     0       0     0      0
      AESMIE               0       0       0      2     0       0     0      0
      AESOC                0       0       0      0     5       0     0      0
      AESOCCD              0       0       0      0     0       2     0      0
      AESOD                0       0       0      0     0       0     0      0
      AESPID               0       0       0      0     0       0     0      0
      AESTDTC              0       0       0      0     0       0     0      0
      AESTDTC_TM           0       0       0      0     0       0     0      0
      AESTRF               0       0       0      0     0       0     0      0
      AETERM               0       0       0      0     0       0     0      0
      AETOXGR              0       0       0      0     0       0     0      0
      DROP                 0       0       0      0     0       0     0      0
      DTHDTC               0       0       0      0     0       0     0      0
      FAORRES              0       0       0      0     0       0     0      0
      QNAM_AESI            0       0       0      0     0       0     0      0
      QNAM_COPD            0       0       0      0     0       0     0      0
      QNAM_ESAM1           0       0       0      0     0       0     0      0
      QNAM_ESAM2           0       0       0      0     0       0     0      0
      QNAM_ESAM3           0       0       0      0     0       0     0      0
      QNAM_EXER            0       0       0      0     0       0     0      0
      QNAM_EXLAB           0       0       0      0     0       0     0      0
      QNAM_EXSAB           0       0       0      0     0       0     0      0
      QNAM_EXSTER          0       0       0      0     0       0     0      0
      SITEID               0       0       0      0     0       0     0      0
      STUDYID              0       0       0      0     0       0     0      0
      SUBJID               0       0       0      0     0       0     0      0
      SUPPAE.QVAL          0       0       0      0     0       0     0      0
                      Reference
    Prediction         AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                  0          0      0      0       0    1      0       0
      AEACNDEV               0          0      0      0       0    0      0       0
      AEACNOTH               0          0      0      0       0    1      0       0
      AECAT                  0          0      0      0       0    0      0       0
      AECONTRT               0          0      0      0       0    0      0       0
      AEDECOD                0          0      0      0       0    0      0       0
      AEDIR                  0          0      0      0       0    0      0       0
      AEDUR                  0          0      0      0       0    0      0       0
      AEENDTC                0          0      0      0       0    0      0       0
      AEENDTC_TM             0          0      0      0       0    0      0       0
      AEENDY                 0          0      0      0       0    0      0       0
      AEENRF                 0          0      0      0       0    0      0       0
      AEENRTPT               0          0      0      0       0    0      0       0
      AEENRTPT; AEENRF       0          0      0      0       0    0      0       0
      AEENTPT                0          0      0      0       0    0      0       0
      AEHLGT                 0          0      0      0       0    0      0       0
      AEHLGTCD               0          0      0      0       0    0      0       0
      AEHLT                  0          0      0      0       0    0      0       0
      AEHLTCD                0          0      0      0       0    0      0       0
      AELAT                  0          0      0      0       0    0      0       0
      AELIFTH                0          0      0      0       0    0      0       0
      AELLT                  0          0      0      0       0    0      0       0
      AELLTCD                0          0      0      0       0    0      0       0
      AELOC                  0          0      0      0       0    0      0       0
      AEMODIFY               0          0      0      0       0    0      0       0
      AEOUT                  0          0      0      0       0    0      0       0
      AEPATT                 0          0      0      0       0    0      0       0
      AEPORTOT               0          0      0      0       0    0      0       0
      AEPRESP                0          0      0      0       0    0      0       0
      AEPTCD                 0          0      0      0       0    0      0       0
      AEREL                  0          0      0      0       0    0      0       0
      AERELNST               0          0      0      0       0    0      0       0
      AESCAN                 0          0      0      0       0    0      0       0
      AESCAT                 0          0      0      0       0    0      0       0
      AESCONG                0          0      0      0       0    0      0       0
      AESDISAB               0          0      0      0       0    0      0       0
      AESDTH                 0          0      0      0       0    0      0       0
      AESER                  0          0      0      0       0    0      0       0
      AESEV                  0          0      0      0       0    0      0       0
      AESHOSP                0          0      0      0       0    1      0       0
      AESLIFE                0          0      0      0       0    0      0       0
      AESMIE                 0          0      0      0       0    0      0       0
      AESOC                  0          0      0      0       0    0      0       0
      AESOCCD                0          0      0      0       0    0      0       0
      AESOD                  0          0      0      0       0    0      0       0
      AESPID                 0          0      0      0       0    0      0       0
      AESTDTC                4          0      0      0       0    0      0       0
      AESTDTC_TM             0          1      0      0       0    0      0       0
      AESTRF                 0          0      1      0       0    0      0       0
      AETERM                 0          0      0      4       0    0      0       0
      AETOXGR                0          0      0      0       0    0      0       0
      DROP                   0          0      0      0       0  196      0       0
      DTHDTC                 0          0      0      0       0    0      0       0
      FAORRES                0          0      0      0       0    0      0       0
      QNAM_AESI              0          0      0      0       0    0      0       0
      QNAM_COPD              0          0      0      0       0    0      0       0
      QNAM_ESAM1             0          0      0      0       0    0      0       0
      QNAM_ESAM2             0          0      0      0       0    0      0       0
      QNAM_ESAM3             0          0      0      0       0    0      0       0
      QNAM_EXER              0          0      0      0       0    0      0       0
      QNAM_EXLAB             0          0      0      0       0    0      0       0
      QNAM_EXSAB             0          0      0      0       0    0      0       0
      QNAM_EXSTER            0          0      0      0       0    0      0       0
      SITEID                 0          0      0      0       0    0      0       0
      STUDYID                0          0      0      0       0    0      0       0
      SUBJID                 0          0      0      0       0    0      0       0
      SUPPAE.QVAL            0          0      0      0       0    0      0       0
                      Reference
    Prediction         QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3
      AEACN                    0         0          0          0          0
      AEACNDEV                 0         0          0          0          0
      AEACNOTH                 0         0          0          0          0
      AECAT                    0         0          0          0          0
      AECONTRT                 0         0          0          0          0
      AEDECOD                  0         0          0          0          0
      AEDIR                    0         0          0          0          0
      AEDUR                    0         0          0          0          0
      AEENDTC                  0         0          0          0          0
      AEENDTC_TM               0         0          0          0          0
      AEENDY                   0         0          0          0          0
      AEENRF                   0         0          0          0          0
      AEENRTPT                 0         0          0          0          0
      AEENRTPT; AEENRF         0         0          0          0          0
      AEENTPT                  0         0          0          0          0
      AEHLGT                   0         0          0          0          0
      AEHLGTCD                 0         0          0          0          0
      AEHLT                    0         0          0          0          0
      AEHLTCD                  0         0          0          0          0
      AELAT                    0         0          0          0          0
      AELIFTH                  0         0          0          0          0
      AELLT                    0         0          0          0          0
      AELLTCD                  0         0          0          0          0
      AELOC                    0         0          0          0          0
      AEMODIFY                 0         0          0          0          0
      AEOUT                    0         0          0          0          0
      AEPATT                   0         0          0          0          0
      AEPORTOT                 0         0          0          0          0
      AEPRESP                  0         0          0          0          0
      AEPTCD                   0         0          0          0          0
      AEREL                    0         0          0          0          0
      AERELNST                 0         0          0          0          0
      AESCAN                   0         0          0          0          0
      AESCAT                   0         0          0          0          0
      AESCONG                  0         0          0          0          0
      AESDISAB                 0         0          0          0          0
      AESDTH                   0         0          0          0          0
      AESER                    0         0          0          0          0
      AESEV                    0         0          0          0          0
      AESHOSP                  0         0          0          0          0
      AESLIFE                  0         0          0          0          0
      AESMIE                   0         0          0          0          0
      AESOC                    0         0          0          0          0
      AESOCCD                  0         0          0          0          0
      AESOD                    0         0          0          0          0
      AESPID                   0         0          0          0          0
      AESTDTC                  0         0          0          0          0
      AESTDTC_TM               0         0          0          0          0
      AESTRF                   0         0          0          0          0
      AETERM                   0         0          0          0          0
      AETOXGR                  0         0          0          0          0
      DROP                     0         0          1          0          0
      DTHDTC                   0         0          0          0          0
      FAORRES                  0         0          0          0          0
      QNAM_AESI                0         0          0          0          0
      QNAM_COPD                0         0          0          0          0
      QNAM_ESAM1               0         0          0          0          0
      QNAM_ESAM2               0         0          0          1          0
      QNAM_ESAM3               0         0          0          0          0
      QNAM_EXER                0         0          0          0          0
      QNAM_EXLAB               0         0          0          0          0
      QNAM_EXSAB               0         0          0          0          0
      QNAM_EXSTER              0         0          0          0          0
      SITEID                   0         0          0          0          0
      STUDYID                  0         0          0          0          0
      SUBJID                   0         0          0          0          0
      SUPPAE.QVAL              0         0          0          0          0
                      Reference
    Prediction         QNAM_EXER QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID
      AEACN                    0          0          0           0      0       0
      AEACNDEV                 0          0          0           0      0       0
      AEACNOTH                 0          0          0           0      0       0
      AECAT                    0          0          0           0      0       0
      AECONTRT                 0          0          0           0      0       0
      AEDECOD                  0          0          0           0      0       0
      AEDIR                    0          0          0           0      0       0
      AEDUR                    0          0          0           0      0       0
      AEENDTC                  0          0          0           0      0       0
      AEENDTC_TM               0          0          0           0      0       0
      AEENDY                   0          0          0           0      0       0
      AEENRF                   0          0          0           0      0       0
      AEENRTPT                 0          0          0           0      0       0
      AEENRTPT; AEENRF         0          0          0           0      0       0
      AEENTPT                  0          0          0           0      0       0
      AEHLGT                   0          0          0           0      0       0
      AEHLGTCD                 0          0          0           0      0       0
      AEHLT                    0          0          0           0      0       0
      AEHLTCD                  0          0          0           0      0       0
      AELAT                    0          0          0           0      0       0
      AELIFTH                  0          0          0           0      0       0
      AELLT                    0          0          0           0      0       0
      AELLTCD                  0          0          0           0      0       0
      AELOC                    0          0          0           0      0       0
      AEMODIFY                 0          0          0           0      0       0
      AEOUT                    0          0          0           0      0       0
      AEPATT                   0          0          0           0      0       0
      AEPORTOT                 0          0          0           0      0       0
      AEPRESP                  0          0          0           0      0       0
      AEPTCD                   0          0          0           0      0       0
      AEREL                    0          0          0           0      0       0
      AERELNST                 0          0          0           0      0       0
      AESCAN                   0          0          0           0      0       0
      AESCAT                   0          0          0           0      0       0
      AESCONG                  0          0          0           0      0       0
      AESDISAB                 0          0          0           0      0       0
      AESDTH                   0          0          0           0      0       0
      AESER                    0          0          0           0      0       0
      AESEV                    0          0          0           0      0       0
      AESHOSP                  0          0          0           0      0       0
      AESLIFE                  0          0          0           0      0       0
      AESMIE                   0          0          0           0      0       0
      AESOC                    0          0          0           0      0       0
      AESOCCD                  0          0          0           0      0       0
      AESOD                    0          0          0           0      0       0
      AESPID                   0          0          0           0      0       0
      AESTDTC                  0          0          0           0      0       0
      AESTDTC_TM               0          0          0           0      0       0
      AESTRF                   0          0          0           0      0       0
      AETERM                   0          0          0           0      0       0
      AETOXGR                  0          0          0           0      0       0
      DROP                     0          0          0           0      0       0
      DTHDTC                   0          0          0           0      0       0
      FAORRES                  0          0          0           0      0       0
      QNAM_AESI                0          0          0           0      0       0
      QNAM_COPD                0          0          0           0      0       0
      QNAM_ESAM1               0          0          0           0      0       0
      QNAM_ESAM2               0          0          0           0      0       0
      QNAM_ESAM3               0          0          0           0      0       0
      QNAM_EXER                0          0          0           0      0       0
      QNAM_EXLAB               0          0          0           0      0       0
      QNAM_EXSAB               0          0          0           0      0       0
      QNAM_EXSTER              0          0          0           0      0       0
      SITEID                   0          0          0           0      4       0
      STUDYID                  0          0          0           0      0       4
      SUBJID                   0          0          0           0      0       0
      SUPPAE.QVAL              0          0          0           0      0       0
                      Reference
    Prediction         SUBJID SUPPAE.QVAL
      AEACN                 0           0
      AEACNDEV              0           0
      AEACNOTH              0           0
      AECAT                 0           0
      AECONTRT              0           0
      AEDECOD               0           0
      AEDIR                 0           0
      AEDUR                 0           0
      AEENDTC               0           0
      AEENDTC_TM            0           0
      AEENDY                0           0
      AEENRF                0           0
      AEENRTPT              0           0
      AEENRTPT; AEENRF      0           0
      AEENTPT               0           0
      AEHLGT                0           0
      AEHLGTCD              0           0
      AEHLT                 0           0
      AEHLTCD               0           0
      AELAT                 0           0
      AELIFTH               0           0
      AELLT                 0           0
      AELLTCD               0           0
      AELOC                 0           0
      AEMODIFY              0           0
      AEOUT                 0           0
      AEPATT                0           0
      AEPORTOT              0           0
      AEPRESP               0           0
      AEPTCD                0           0
      AEREL                 0           0
      AERELNST              0           0
      AESCAN                0           0
      AESCAT                0           0
      AESCONG               0           0
      AESDISAB              0           0
      AESDTH                0           0
      AESER                 0           0
      AESEV                 0           0
      AESHOSP               0           0
      AESLIFE               0           0
      AESMIE                0           0
      AESOC                 0           0
      AESOCCD               0           0
      AESOD                 0           0
      AESPID                0           0
      AESTDTC               0           0
      AESTDTC_TM            0           0
      AESTRF                0           0
      AETERM                0           0
      AETOXGR               0           0
      DROP                  0           0
      DTHDTC                0           0
      FAORRES               0           0
      QNAM_AESI             0           0
      QNAM_COPD             0           0
      QNAM_ESAM1            0           0
      QNAM_ESAM2            0           0
      QNAM_ESAM3            0           0
      QNAM_EXER             0           0
      QNAM_EXLAB            0           0
      QNAM_EXSAB            0           0
      QNAM_EXSTER           0           0
      SITEID                0           0
      STUDYID               0           0
      SUBJID                4           0
      SUPPAE.QVAL           0           0
    
    Overall Statistics
                                              
                   Accuracy : 0.9575          
                     95% CI : (0.9284, 0.9772)
        No Information Rate : 0.6503          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.925           
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity              0.750000              NA              NA           NA
    Specificity              0.996689               1        0.996732            1
    Pos Pred Value           0.750000              NA              NA           NA
    Neg Pred Value           0.996689              NA              NA           NA
    Prevalence               0.013072               0        0.000000            0
    Detection Rate           0.009804               0        0.000000            0
    Detection Prevalence     0.013072               0        0.003268            0
    Balanced Accuracy        0.873344              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000       1.000000           NA           NA
    Specificity                 1.000000       1.000000            1            1
    Pos Pred Value              1.000000       1.000000           NA           NA
    Neg Pred Value              1.000000       1.000000           NA           NA
    Prevalence                  0.006536       0.009804            0            0
    Detection Rate              0.006536       0.009804            0            0
    Detection Prevalence        0.006536       0.009804            0            0
    Balanced Accuracy           1.000000       1.000000           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                0.500000          1.000000            NA
    Specificity                1.000000          1.000000             1
    Pos Pred Value             1.000000          1.000000            NA
    Neg Pred Value             0.993421          1.000000            NA
    Prevalence                 0.013072          0.003268             0
    Detection Rate             0.006536          0.003268             0
    Detection Prevalence       0.006536          0.003268             0
    Balanced Accuracy          0.750000          1.000000            NA
                         Class: AEENRF Class: AEENRTPT Class: AEENRTPT; AEENRF
    Sensitivity                     NA         1.00000                      NA
    Specificity                      1         1.00000                       1
    Pos Pred Value                  NA         1.00000                      NA
    Neg Pred Value                  NA         1.00000                      NA
    Prevalence                       0         0.01307                       0
    Detection Rate                   0         0.01307                       0
    Detection Prevalence             0         0.01307                       0
    Balanced Accuracy               NA         1.00000                      NA
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA       1.00000        0.750000     0.750000
    Specificity                       1       0.99669        1.000000     1.000000
    Pos Pred Value                   NA       0.80000        1.000000     1.000000
    Neg Pred Value                   NA       1.00000        0.996700     0.996700
    Prevalence                        0       0.01307        0.013072     0.013072
    Detection Rate                    0       0.01307        0.009804     0.009804
    Detection Prevalence              0       0.01634        0.009804     0.009804
    Balanced Accuracy                NA       0.99834        0.875000     0.875000
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                 1.00000           NA       1.000000      1.00000
    Specificity                 0.99338            1       1.000000      1.00000
    Pos Pred Value              0.66667           NA       1.000000      1.00000
    Neg Pred Value              1.00000           NA       1.000000      1.00000
    Prevalence                  0.01307            0       0.006536      0.01307
    Detection Rate              0.01307            0       0.006536      0.01307
    Detection Prevalence        0.01961            0       0.006536      0.01307
    Balanced Accuracy           0.99669           NA       1.000000      1.00000
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                1.000000           NA              NA      1.00000
    Specificity                1.000000            1               1      1.00000
    Pos Pred Value             1.000000           NA              NA      1.00000
    Neg Pred Value             1.000000           NA              NA      1.00000
    Prevalence                 0.003268            0               0      0.01307
    Detection Rate             0.003268            0               0      0.01307
    Detection Prevalence       0.003268            0               0      0.01307
    Balanced Accuracy          1.000000           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity               0.750000              NA             NA      0.750000
    Specificity               1.000000               1              1      1.000000
    Pos Pred Value            1.000000              NA             NA      1.000000
    Neg Pred Value            0.996700              NA             NA      0.996700
    Prevalence                0.013072               0              0      0.013072
    Detection Rate            0.009804               0              0      0.009804
    Detection Prevalence      0.009804               0              0      0.009804
    Balanced Accuracy         0.875000              NA             NA      0.875000
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity               1.00000              NA            NA            NA
    Specificity               1.00000               1             1             1
    Pos Pred Value            1.00000              NA            NA            NA
    Neg Pred Value            1.00000              NA            NA            NA
    Prevalence                0.01307               0             0             0
    Detection Rate            0.01307               0             0             0
    Detection Prevalence      0.01307               0             0             0
    Balanced Accuracy         1.00000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        1.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     1.000000
    Pos Pred Value             1.000000        1.000000           NaN     1.000000
    Neg Pred Value             1.000000        1.000000      0.996732     0.996700
    Prevalence                 0.006536        0.009804      0.003268     0.013072
    Detection Rate             0.006536        0.009804      0.000000     0.009804
    Detection Prevalence       0.006536        0.009804      0.000000     0.009804
    Balanced Accuracy          1.000000        1.000000      0.500000     0.875000
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       1.000000             NA      1.000000
    Specificity               1.00000       0.996700              1      1.000000
    Pos Pred Value            1.00000       0.750000             NA      1.000000
    Neg Pred Value            1.00000       1.000000             NA      1.000000
    Prevalence                0.01307       0.009804              0      0.006536
    Detection Rate            0.01307       0.009804              0      0.006536
    Detection Prevalence      0.01307       0.013072              0      0.006536
    Balanced Accuracy         1.00000       0.998350             NA      1.000000
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity               1.00000       1.000000           NA            NA
    Specificity               1.00000       1.000000            1             1
    Pos Pred Value            1.00000       1.000000           NA            NA
    Neg Pred Value            1.00000       1.000000           NA            NA
    Prevalence                0.01634       0.006536            0             0
    Detection Rate            0.01634       0.006536            0             0
    Detection Prevalence      0.01634       0.006536            0             0
    Balanced Accuracy         1.00000       1.000000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                 1.00000          1.000000      1.000000
    Specificity                 1.00000          1.000000      1.000000
    Pos Pred Value              1.00000          1.000000      1.000000
    Neg Pred Value              1.00000          1.000000      1.000000
    Prevalence                  0.01307          0.003268      0.003268
    Detection Rate              0.01307          0.003268      0.003268
    Detection Prevalence        0.01307          0.003268      0.003268
    Balanced Accuracy           1.00000          1.000000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9849            NA
    Specificity                1.00000              1      0.9439             1
    Pos Pred Value             1.00000             NA      0.9703            NA
    Neg Pred Value             1.00000             NA      0.9712            NA
    Prevalence                 0.01307              0      0.6503             0
    Detection Rate             0.01307              0      0.6405             0
    Detection Prevalence       0.01307              0      0.6601             0
    Balanced Accuracy          1.00000             NA      0.9644            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          1.000000                NA
    Specificity                   1.000000          1.000000                 1
    Pos Pred Value                     NaN          1.000000                NA
    Neg Pred Value                0.996732          1.000000                NA
    Prevalence                    0.003268          0.003268                 0
    Detection Rate                0.000000          0.003268                 0
    Detection Prevalence          0.000000          0.003268                 0
    Balanced Accuracy             0.500000          1.000000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000        1.00000
    Specificity                           1       1.00000        0.99669
    Pos Pred Value                       NA       1.00000        0.80000
    Neg Pred Value                       NA       1.00000        1.00000
    Prevalence                            0       0.01307        0.01307
    Detection Rate                        0       0.01307        0.01307
    Detection Prevalence                  0       0.01307        0.01634
    Balanced Accuracy                    NA       1.00000        0.99834
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000                  1
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01307                  0
    Detection Rate             0.01307                  0
    Detection Prevalence       0.01307                  0
    Balanced Accuracy          1.00000                 NA


**SVM**


```R
prediction <-predict(svm_mod1, newdata=test)
cm.svm_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.svm_mod
```


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              4        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        2       0     0     0
      AEDECOD            0        0        0     0        0       4     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       0     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               0        0        0     0        0       0     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              4          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        4
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 0          0      0      0              0        0
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        0     0       0     0       0     0
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      4        0     1       0     0       0     0
      AEHLGTCD             0      0        3     0       0     0       0     0
      AEHLT                0      0        0     3       0     0       0     0
      AEHLTCD              0      0        1     0       4     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       2     0
      AELLT                0      0        0     0       0     0       0     4
      AELLTCD              0      0        0     0       0     0       0     0
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       0     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      0        0     0       0     0       0     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      0
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      0
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                0     0        0     0      0        0       0      0
      AELLTCD              4     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      4        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               0     0        0     0      0        0       0      3
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 0     0        0     0      0        0       0      1
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      0     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              4        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       2        0      0     0
      AESDISAB           0        0      0      0       0        3      0     0
      AESDTH             0        0      0      0       0        0      0     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               0        0      0      0       0        0      1     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      0     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     0       0     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       0     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              4       0       0      0     0       0     0      0
      AESHOSP            0       2       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     5       0     0      0
      AESOCCD            0       0       0      0     0       5     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               0       1       0      0     0       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       0       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      0       0    0      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    0      0       0
      AEENDTC_TM           0          0      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    0      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    0      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    2      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              4          0      0      0       0    0      0       0
      AESTDTC_TM           0          1      0      0       0    0      0       0
      AESTRF               0          0      1      0       0    0      0       0
      AETERM               0          0      0      4       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 0          0      0      0       0  201      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          1          0          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          0          0          0         0
      QNAM_ESAM2             0         0          0          1          0         0
      QNAM_ESAM3             0         0          0          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      0       0      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      4       1      0
      STUDYID                 0          0           0      0       3      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                             
                   Accuracy : 0.9654         
                     95% CI : (0.939, 0.9826)
        No Information Rate : 0.6415         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.9404         
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity               1.00000              NA              NA           NA
    Specificity               0.99682               1               1            1
    Pos Pred Value            0.80000              NA              NA           NA
    Neg Pred Value            1.00000              NA              NA           NA
    Prevalence                0.01258               0               0            0
    Detection Rate            0.01258               0               0            0
    Detection Prevalence      0.01572               0               0            0
    Balanced Accuracy         0.99841              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000        1.00000           NA           NA
    Specificity                 1.000000        1.00000            1            1
    Pos Pred Value              1.000000        1.00000           NA           NA
    Neg Pred Value              1.000000        1.00000           NA           NA
    Prevalence                  0.006289        0.01258            0            0
    Detection Rate              0.006289        0.01258            0            0
    Detection Prevalence        0.006289        0.01258            0            0
    Balanced Accuracy           1.000000        1.00000           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                 1.00000          1.000000            NA
    Specificity                 1.00000          1.000000             1
    Pos Pred Value              1.00000          1.000000            NA
    Neg Pred Value              1.00000          1.000000            NA
    Prevalence                  0.01258          0.003145             0
    Detection Rate              0.01258          0.003145             0
    Detection Prevalence        0.01258          0.003145             0
    Balanced Accuracy           1.00000          1.000000            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         1.00000
    Specificity                      1                     1         1.00000
    Pos Pred Value                  NA                    NA         1.00000
    Neg Pred Value                  NA                    NA         1.00000
    Prevalence                       0                     0         0.01258
    Detection Rate                   0                     0         0.01258
    Detection Prevalence             0                     0         0.01258
    Balanced Accuracy               NA                    NA         1.00000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA       1.00000        0.750000     0.750000
    Specificity                       1       0.99682        1.000000     1.000000
    Pos Pred Value                   NA       0.80000        1.000000     1.000000
    Neg Pred Value                   NA       1.00000        0.996825     0.996825
    Prevalence                        0       0.01258        0.012579     0.012579
    Detection Rate                    0       0.01258        0.009434     0.009434
    Detection Prevalence              0       0.01572        0.009434     0.009434
    Balanced Accuracy                NA       0.99841        0.875000     0.875000
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                 1.00000           NA       1.000000      1.00000
    Specificity                 0.99682            1       1.000000      1.00000
    Pos Pred Value              0.80000           NA       1.000000      1.00000
    Neg Pred Value              1.00000           NA       1.000000      1.00000
    Prevalence                  0.01258            0       0.006289      0.01258
    Detection Rate              0.01258            0       0.006289      0.01258
    Detection Prevalence        0.01572            0       0.006289      0.01258
    Balanced Accuracy           0.99841           NA       1.000000      1.00000
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                 1.00000           NA              NA      1.00000
    Specificity                 1.00000            1               1      1.00000
    Pos Pred Value              1.00000           NA              NA      1.00000
    Neg Pred Value              1.00000           NA              NA      1.00000
    Prevalence                  0.01258            0               0      0.01258
    Detection Rate              0.01258            0               0      0.01258
    Detection Prevalence        0.01258            0               0      0.01258
    Balanced Accuracy           1.00000           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity                1.00000              NA             NA      0.750000
    Specificity                1.00000               1              1      1.000000
    Pos Pred Value             1.00000              NA             NA      1.000000
    Neg Pred Value             1.00000              NA             NA      0.996825
    Prevalence                 0.01258               0              0      0.012579
    Detection Rate             0.01258               0              0      0.009434
    Detection Prevalence       0.01258               0              0      0.009434
    Balanced Accuracy          1.00000              NA             NA      0.875000
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity               1.00000              NA            NA            NA
    Specificity               1.00000               1             1             1
    Pos Pred Value            1.00000              NA            NA            NA
    Neg Pred Value            1.00000              NA            NA            NA
    Prevalence                0.01258               0             0             0
    Detection Rate            0.01258               0             0             0
    Detection Prevalence      0.01258               0             0             0
    Balanced Accuracy         1.00000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        1.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     1.000000
    Pos Pred Value             1.000000        1.000000           NaN     1.000000
    Neg Pred Value             1.000000        1.000000      0.996855     0.996825
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.006289        0.009434      0.000000     0.009434
    Detection Prevalence       0.006289        0.009434      0.000000     0.009434
    Balanced Accuracy          1.000000        1.000000      0.500000     0.875000
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       0.666667             NA      1.000000
    Specificity               1.00000       1.000000              1      0.993671
    Pos Pred Value            1.00000       1.000000             NA      0.500000
    Neg Pred Value            1.00000       0.996835             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.01258       0.006289              0      0.006289
    Detection Prevalence      0.01258       0.006289              0      0.012579
    Balanced Accuracy         1.00000       0.833333             NA      0.996835
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity               1.00000        1.00000           NA            NA
    Specificity               1.00000        1.00000            1             1
    Pos Pred Value            1.00000        1.00000           NA            NA
    Neg Pred Value            1.00000        1.00000           NA            NA
    Prevalence                0.01572        0.01572            0             0
    Detection Rate            0.01572        0.01572            0             0
    Detection Prevalence      0.01572        0.01572            0             0
    Balanced Accuracy         1.00000        1.00000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                 1.00000          1.000000      1.000000
    Specificity                 1.00000          1.000000      1.000000
    Pos Pred Value              1.00000          1.000000      1.000000
    Neg Pred Value              1.00000          1.000000      1.000000
    Prevalence                  0.01258          0.003145      0.003145
    Detection Rate              0.01258          0.003145      0.003145
    Detection Prevalence        0.01258          0.003145      0.003145
    Balanced Accuracy           1.00000          1.000000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9853            NA
    Specificity                1.00000              1      0.9561             1
    Pos Pred Value             1.00000             NA      0.9757            NA
    Neg Pred Value             1.00000             NA      0.9732            NA
    Prevalence                 0.01258              0      0.6415             0
    Detection Rate             0.01258              0      0.6321             0
    Detection Prevalence       0.01258              0      0.6478             0
    Balanced Accuracy          1.00000             NA      0.9707            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          1.000000                NA
    Specificity                   1.000000          1.000000                 1
    Pos Pred Value                     NaN          1.000000                NA
    Neg Pred Value                0.996855          1.000000                NA
    Prevalence                    0.003145          0.003145                 0
    Detection Rate                0.000000          0.003145                 0
    Detection Prevalence          0.000000          0.003145                 0
    Balanced Accuracy             0.500000          1.000000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000       0.750000
    Specificity                           1       0.99682       1.000000
    Pos Pred Value                       NA       0.80000       1.000000
    Neg Pred Value                       NA       1.00000       0.996825
    Prevalence                            0       0.01258       0.012579
    Detection Rate                        0       0.01258       0.009434
    Detection Prevalence                  0       0.01572       0.009434
    Balanced Accuracy                    NA       0.99841       0.875000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000                  1
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258                  0
    Detection Rate             0.01258                  0
    Detection Prevalence       0.01258                  0
    Balanced Accuracy          1.00000                 NA


**CART**


```R
prediction <-predict(cart_mod3, newdata=test)
cm.cart_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.cart_mod
```


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              4        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        2       0     0     0
      AEDECOD            0        0        0     0        0       4     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       0     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               0        0        0     0        0       0     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              2          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        4
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 2          0      0      0              0        0
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        1     0       1     0       0     1
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      1        0     0       0     0       0     0
      AEHLGTCD             0      0        0     0       0     0       0     0
      AEHLT                0      2        0     3       0     0       0     0
      AEHLTCD              0      0        2     1       1     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       2     0
      AELLT                0      0        0     0       0     0       0     2
      AELLTCD              0      0        1     0       0     0       0     1
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       2     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      1        0     0       0     0       0     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      2
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      0
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                1     0        0     0      0        0       0      0
      AELLTCD              1     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      3        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               2     0        0     0      0        0       0      2
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 0     0        0     0      1        0       0      0
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      0     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              3        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       2        0      0     0
      AESDISAB           0        0      0      0       0        3      0     0
      AESDTH             0        0      0      0       0        0      0     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               1        0      0      0       0        0      0     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      1     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     1       1     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       2     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              4       0       0      0     0       0     0      0
      AESHOSP            0       2       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     1       0     0      0
      AESOCCD            0       0       0      0     3       2     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               0       0       0      0     0       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       1       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      1       0    1      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    0      0       0
      AEENDTC_TM           0          1      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    1      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    0      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    2      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              3          0      0      0       0    1      0       0
      AESTDTC_TM           0          0      0      0       0    0      0       0
      AESTRF               0          0      1      0       0    0      0       0
      AETERM               0          0      0      3       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 1          0      0      0       0  198      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          1          1          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          0          0          0         0
      QNAM_ESAM2             0         0          0          0          0         0
      QNAM_ESAM3             0         0          0          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      0       1      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      4       0      0
      STUDYID                 0          0           0      0       3      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                              
                   Accuracy : 0.8616          
                     95% CI : (0.8187, 0.8976)
        No Information Rate : 0.6415          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.76            
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity               1.00000              NA              NA           NA
    Specificity               0.99682               1               1            1
    Pos Pred Value            0.80000              NA              NA           NA
    Neg Pred Value            1.00000              NA              NA           NA
    Prevalence                0.01258               0               0            0
    Detection Rate            0.01258               0               0            0
    Detection Prevalence      0.01572               0               0            0
    Balanced Accuracy         0.99841              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000        1.00000           NA           NA
    Specificity                 1.000000        0.97134            1            1
    Pos Pred Value              1.000000        0.30769           NA           NA
    Neg Pred Value              1.000000        1.00000           NA           NA
    Prevalence                  0.006289        0.01258            0            0
    Detection Rate              0.006289        0.01258            0            0
    Detection Prevalence        0.006289        0.04088            0            0
    Balanced Accuracy           1.000000        0.98567           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                0.500000          1.000000            NA
    Specificity                1.000000          0.996845             1
    Pos Pred Value             1.000000          0.500000            NA
    Neg Pred Value             0.993671          1.000000            NA
    Prevalence                 0.012579          0.003145             0
    Detection Rate             0.006289          0.003145             0
    Detection Prevalence       0.006289          0.006289             0
    Balanced Accuracy          0.750000          0.998423            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         1.00000
    Specificity                      1                     1         1.00000
    Pos Pred Value                  NA                    NA         1.00000
    Neg Pred Value                  NA                    NA         1.00000
    Prevalence                       0                     0         0.01258
    Detection Rate                   0                     0         0.01258
    Detection Prevalence             0                     0         0.01258
    Balanced Accuracy               NA                    NA         1.00000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA      0.250000         0.00000     0.750000
    Specificity                       1      1.000000         1.00000     0.993631
    Pos Pred Value                   NA      1.000000             NaN     0.600000
    Neg Pred Value                   NA      0.990536         0.98742     0.996805
    Prevalence                        0      0.012579         0.01258     0.012579
    Detection Rate                    0      0.003145         0.00000     0.009434
    Detection Prevalence              0      0.003145         0.00000     0.015723
    Balanced Accuracy                NA      0.625000         0.50000     0.871815
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                0.250000           NA       1.000000     0.500000
    Specificity                0.990446            1       1.000000     0.996815
    Pos Pred Value             0.250000           NA       1.000000     0.666667
    Neg Pred Value             0.990446           NA       1.000000     0.993651
    Prevalence                 0.012579            0       0.006289     0.012579
    Detection Rate             0.003145            0       0.006289     0.006289
    Detection Prevalence       0.012579            0       0.006289     0.009434
    Balanced Accuracy          0.620223           NA       1.000000     0.748408
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                0.250000           NA              NA      1.00000
    Specificity                0.993631            1               1      0.99682
    Pos Pred Value             0.333333           NA              NA      0.80000
    Neg Pred Value             0.990476           NA              NA      1.00000
    Prevalence                 0.012579            0               0      0.01258
    Detection Rate             0.003145            0               0      0.01258
    Detection Prevalence       0.009434            0               0      0.01572
    Balanced Accuracy          0.621815           NA              NA      0.99841
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity               0.750000              NA             NA      0.500000
    Specificity               1.000000               1              1      0.980892
    Pos Pred Value            1.000000              NA             NA      0.250000
    Neg Pred Value            0.996825              NA             NA      0.993548
    Prevalence                0.012579               0              0      0.012579
    Detection Rate            0.009434               0              0      0.006289
    Detection Prevalence      0.009434               0              0      0.025157
    Balanced Accuracy         0.875000              NA             NA      0.740446
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity              0.750000              NA            NA            NA
    Specificity              1.000000               1             1             1
    Pos Pred Value           1.000000              NA            NA            NA
    Neg Pred Value           0.996825              NA            NA            NA
    Prevalence               0.012579               0             0             0
    Detection Rate           0.009434               0             0             0
    Detection Prevalence     0.009434               0             0             0
    Balanced Accuracy        0.875000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        1.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     1.000000
    Pos Pred Value             1.000000        1.000000           NaN     1.000000
    Neg Pred Value             1.000000        1.000000      0.996855     0.996825
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.006289        0.009434      0.000000     0.009434
    Detection Prevalence       0.006289        0.009434      0.000000     0.009434
    Balanced Accuracy          1.000000        1.000000      0.500000     0.875000
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       0.666667             NA      1.000000
    Specificity               1.00000       1.000000              1      0.993671
    Pos Pred Value            1.00000       1.000000             NA      0.500000
    Neg Pred Value            1.00000       0.996835             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.01258       0.006289              0      0.006289
    Detection Prevalence      0.01258       0.006289              0      0.012579
    Balanced Accuracy         1.00000       0.833333             NA      0.996835
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity              0.200000       0.400000           NA            NA
    Specificity              1.000000       0.990415            1             1
    Pos Pred Value           1.000000       0.400000           NA            NA
    Neg Pred Value           0.987382       0.990415           NA            NA
    Prevalence               0.015723       0.015723            0             0
    Detection Rate           0.003145       0.006289            0             0
    Detection Prevalence     0.003145       0.015723            0             0
    Balanced Accuracy        0.600000       0.695208           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                0.750000          0.000000      1.000000
    Specificity                0.996815          1.000000      1.000000
    Pos Pred Value             0.750000               NaN      1.000000
    Neg Pred Value             0.996815          0.996855      1.000000
    Prevalence                 0.012579          0.003145      0.003145
    Detection Rate             0.009434          0.000000      0.003145
    Detection Prevalence       0.012579          0.000000      0.003145
    Balanced Accuracy          0.873408          0.500000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity               0.750000             NA      0.9706            NA
    Specificity               1.000000              1      0.9123             1
    Pos Pred Value            1.000000             NA      0.9519            NA
    Neg Pred Value            0.996825             NA      0.9455            NA
    Prevalence                0.012579              0      0.6415             0
    Detection Rate            0.009434              0      0.6226             0
    Detection Prevalence      0.009434              0      0.6541             0
    Balanced Accuracy         0.875000             NA      0.9414            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          0.000000                NA
    Specificity                   1.000000          1.000000                 1
    Pos Pred Value                     NaN               NaN                NA
    Neg Pred Value                0.996855          0.996855                NA
    Prevalence                    0.003145          0.003145                 0
    Detection Rate                0.000000          0.000000                 0
    Detection Prevalence          0.000000          0.000000                 0
    Balanced Accuracy             0.500000          0.500000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000       0.750000
    Specificity                           1       1.00000       1.000000
    Pos Pred Value                       NA       1.00000       1.000000
    Neg Pred Value                       NA       1.00000       0.996825
    Prevalence                            0       0.01258       0.012579
    Detection Rate                        0       0.01258       0.009434
    Detection Prevalence                  0       0.01258       0.009434
    Balanced Accuracy                    NA       1.00000       0.875000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000           0.993711
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258           0.000000
    Detection Rate             0.01258           0.000000
    Detection Prevalence       0.01258           0.006289
    Balanced Accuracy          1.00000                 NA


**bagged CART**


```R
prediction <-predict(bg.100, newdata=test)
cm.bg_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.bg_mod
```


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              4        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        2       0     0     0
      AEDECOD            0        0        0     0        0       4     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       0     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               0        0        0     0        0       0     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              2          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        4
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 2          0      0      0              0        0
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        0     0       0     0       0     0
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      3        0     1       0     0       0     0
      AEHLGTCD             0      0        3     0       0     0       0     0
      AEHLT                0      0        0     3       0     0       0     0
      AEHLTCD              0      0        1     0       4     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       2     0
      AELLT                0      0        0     0       0     0       0     4
      AELLTCD              0      0        0     0       0     0       0     0
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       0     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      1        0     0       0     0       0     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      0
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      0
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                0     0        0     0      0        0       0      0
      AELLTCD              4     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      3        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               0     0        0     0      0        0       0      4
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 0     0        0     0      1        0       0      0
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      0     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              3        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       2        0      0     0
      AESDISAB           0        0      0      0       0        3      0     0
      AESDTH             0        0      0      0       0        0      0     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               1        0      0      0       0        0      1     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      0     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     0       0     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       0     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              4       0       0      0     0       0     0      0
      AESHOSP            0       2       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     5       0     0      0
      AESOCCD            0       0       0      0     0       5     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               0       0       0      0     0       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       1       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      0       0    0      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    0      0       0
      AEENDTC_TM           0          0      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    0      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    1      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    2      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              3          0      0      0       0    0      0       0
      AESTDTC_TM           0          1      0      0       0    0      0       0
      AESTRF               0          0      1      0       0    0      0       0
      AETERM               0          0      0      4       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 1          0      0      0       0  200      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          0          0          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          0          0          0         0
      QNAM_ESAM2             0         0          0          1          0         0
      QNAM_ESAM3             0         0          1          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      0       1      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      4       0      0
      STUDYID                 0          0           0      0       3      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                              
                   Accuracy : 0.9465          
                     95% CI : (0.9158, 0.9686)
        No Information Rate : 0.6415          
        P-Value [Acc > NIR] : < 2.2e-16       
                                              
                      Kappa : 0.9069          
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity               1.00000              NA              NA           NA
    Specificity               0.99682               1               1            1
    Pos Pred Value            0.80000              NA              NA           NA
    Neg Pred Value            1.00000              NA              NA           NA
    Prevalence                0.01258               0               0            0
    Detection Rate            0.01258               0               0            0
    Detection Prevalence      0.01572               0               0            0
    Balanced Accuracy         0.99841              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000        1.00000           NA           NA
    Specificity                 1.000000        1.00000            1            1
    Pos Pred Value              1.000000        1.00000           NA           NA
    Neg Pred Value              1.000000        1.00000           NA           NA
    Prevalence                  0.006289        0.01258            0            0
    Detection Rate              0.006289        0.01258            0            0
    Detection Prevalence        0.006289        0.01258            0            0
    Balanced Accuracy           1.000000        1.00000           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                0.500000          1.000000            NA
    Specificity                1.000000          1.000000             1
    Pos Pred Value             1.000000          1.000000            NA
    Neg Pred Value             0.993671          1.000000            NA
    Prevalence                 0.012579          0.003145             0
    Detection Rate             0.006289          0.003145             0
    Detection Prevalence       0.006289          0.003145             0
    Balanced Accuracy          0.750000          1.000000            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         1.00000
    Specificity                      1                     1         1.00000
    Pos Pred Value                  NA                    NA         1.00000
    Neg Pred Value                  NA                    NA         1.00000
    Prevalence                       0                     0         0.01258
    Detection Rate                   0                     0         0.01258
    Detection Prevalence             0                     0         0.01258
    Balanced Accuracy               NA                    NA         1.00000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA      0.750000        0.750000     0.750000
    Specificity                       1      0.996815        1.000000     1.000000
    Pos Pred Value                   NA      0.750000        1.000000     1.000000
    Neg Pred Value                   NA      0.996815        0.996825     0.996825
    Prevalence                        0      0.012579        0.012579     0.012579
    Detection Rate                    0      0.009434        0.009434     0.009434
    Detection Prevalence              0      0.012579        0.009434     0.009434
    Balanced Accuracy                NA      0.873408        0.875000     0.875000
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                 1.00000           NA       1.000000      1.00000
    Specificity                 0.99682            1       1.000000      1.00000
    Pos Pred Value              0.80000           NA       1.000000      1.00000
    Neg Pred Value              1.00000           NA       1.000000      1.00000
    Prevalence                  0.01258            0       0.006289      0.01258
    Detection Rate              0.01258            0       0.006289      0.01258
    Detection Prevalence        0.01572            0       0.006289      0.01258
    Balanced Accuracy           0.99841           NA       1.000000      1.00000
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                 1.00000           NA              NA      1.00000
    Specificity                 1.00000            1               1      1.00000
    Pos Pred Value              1.00000           NA              NA      1.00000
    Neg Pred Value              1.00000           NA              NA      1.00000
    Prevalence                  0.01258            0               0      0.01258
    Detection Rate              0.01258            0               0      0.01258
    Detection Prevalence        0.01258            0               0      0.01258
    Balanced Accuracy           1.00000           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity               0.750000              NA             NA       1.00000
    Specificity               1.000000               1              1       1.00000
    Pos Pred Value            1.000000              NA             NA       1.00000
    Neg Pred Value            0.996825              NA             NA       1.00000
    Prevalence                0.012579               0              0       0.01258
    Detection Rate            0.009434               0              0       0.01258
    Detection Prevalence      0.009434               0              0       0.01258
    Balanced Accuracy         0.875000              NA             NA       1.00000
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity              0.750000              NA            NA            NA
    Specificity              1.000000               1             1             1
    Pos Pred Value           1.000000              NA            NA            NA
    Neg Pred Value           0.996825              NA            NA            NA
    Prevalence               0.012579               0             0             0
    Detection Rate           0.009434               0             0             0
    Detection Prevalence     0.009434               0             0             0
    Balanced Accuracy        0.875000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        1.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     0.996815
    Pos Pred Value             1.000000        1.000000           NaN     0.750000
    Neg Pred Value             1.000000        1.000000      0.996855     0.996815
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.006289        0.009434      0.000000     0.009434
    Detection Prevalence       0.006289        0.009434      0.000000     0.012579
    Balanced Accuracy          1.000000        1.000000      0.500000     0.873408
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       0.666667             NA      1.000000
    Specificity               1.00000       1.000000              1      0.993671
    Pos Pred Value            1.00000       1.000000             NA      0.500000
    Neg Pred Value            1.00000       0.996835             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.01258       0.006289              0      0.006289
    Detection Prevalence      0.01258       0.006289              0      0.012579
    Balanced Accuracy         1.00000       0.833333             NA      0.996835
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity               1.00000        1.00000           NA            NA
    Specificity               1.00000        1.00000            1             1
    Pos Pred Value            1.00000        1.00000           NA            NA
    Neg Pred Value            1.00000        1.00000           NA            NA
    Prevalence                0.01572        0.01572            0             0
    Detection Rate            0.01572        0.01572            0             0
    Detection Prevalence      0.01572        0.01572            0             0
    Balanced Accuracy         1.00000        1.00000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                0.750000          1.000000      1.000000
    Specificity                1.000000          1.000000      1.000000
    Pos Pred Value             1.000000          1.000000      1.000000
    Neg Pred Value             0.996825          1.000000      1.000000
    Prevalence                 0.012579          0.003145      0.003145
    Detection Rate             0.009434          0.003145      0.003145
    Detection Prevalence       0.009434          0.003145      0.003145
    Balanced Accuracy          0.875000          1.000000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9804            NA
    Specificity                1.00000              1      0.9211             1
    Pos Pred Value             1.00000             NA      0.9569            NA
    Neg Pred Value             1.00000             NA      0.9633            NA
    Prevalence                 0.01258              0      0.6415             0
    Detection Rate             0.01258              0      0.6289             0
    Detection Prevalence       0.01258              0      0.6572             0
    Balanced Accuracy          1.00000             NA      0.9507            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          1.000000                NA
    Specificity                   1.000000          1.000000          0.996855
    Pos Pred Value                     NaN          1.000000                NA
    Neg Pred Value                0.996855          1.000000                NA
    Prevalence                    0.003145          0.003145          0.000000
    Detection Rate                0.000000          0.003145          0.000000
    Detection Prevalence          0.000000          0.003145          0.003145
    Balanced Accuracy             0.500000          1.000000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000       0.750000
    Specificity                           1       1.00000       1.000000
    Pos Pred Value                       NA       1.00000       1.000000
    Neg Pred Value                       NA       1.00000       0.996825
    Prevalence                            0       0.01258       0.012579
    Detection Rate                        0       0.01258       0.009434
    Detection Prevalence                  0       0.01258       0.009434
    Balanced Accuracy                    NA       1.00000       0.875000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000           0.996855
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258           0.000000
    Detection Rate             0.01258           0.000000
    Detection Prevalence       0.01258           0.003145
    Balanced Accuracy          1.00000                 NA


**random forest**


```R
prediction <-predict(rf.500, newdata=test)
cm.rf_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.rf_mod
```


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              4        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        2       0     0     0
      AEDECOD            0        0        0     0        0       3     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       0     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               0        0        0     0        0       1     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              4          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        4
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 0          0      0      0              0        0
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        0     0       0     0       0     0
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      4        0     1       0     0       0     0
      AEHLGTCD             0      0        3     0       0     0       0     0
      AEHLT                0      0        0     3       0     0       0     0
      AEHLTCD              0      0        1     0       4     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       2     0
      AELLT                0      0        0     0       0     0       0     4
      AELLTCD              0      0        0     0       0     0       0     0
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       0     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      0        0     0       0     0       0     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      0
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      1
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                0     0        0     0      0        0       0      0
      AELLTCD              4     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      4        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               0     0        0     0      0        0       0      3
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 0     0        0     0      0        0       0      0
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      0     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              4        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       2        0      0     0
      AESDISAB           0        0      0      0       0        3      0     0
      AESDTH             0        0      0      0       0        0      0     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               0        0      0      0       0        0      0     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      1     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     0       0     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       0     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              4       0       0      0     0       0     0      0
      AESHOSP            0       2       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     5       0     0      0
      AESOCCD            0       0       0      0     0       5     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               0       0       0      0     0       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       1       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      0       0    0      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    0      0       0
      AEENDTC_TM           0          0      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    0      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    1      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    2      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              4          0      0      0       0    0      0       0
      AESTDTC_TM           0          1      0      0       0    0      0       0
      AESTRF               0          0      1      0       0    0      0       0
      AETERM               0          0      0      4       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 0          0      0      0       0  200      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          0          0          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          0          0          0         0
      QNAM_ESAM2             0         0          0          1          0         0
      QNAM_ESAM3             0         0          1          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      0       0      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      4       0      0
      STUDYID                 0          0           0      0       4      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                             
                   Accuracy : 0.9623         
                     95% CI : (0.935, 0.9804)
        No Information Rate : 0.6415         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.9359         
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity               1.00000              NA              NA           NA
    Specificity               0.99682               1               1            1
    Pos Pred Value            0.80000              NA              NA           NA
    Neg Pred Value            1.00000              NA              NA           NA
    Prevalence                0.01258               0               0            0
    Detection Rate            0.01258               0               0            0
    Detection Prevalence      0.01572               0               0            0
    Balanced Accuracy         0.99841              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000       0.750000           NA           NA
    Specificity                 1.000000       1.000000            1            1
    Pos Pred Value              1.000000       1.000000           NA           NA
    Neg Pred Value              1.000000       0.996825           NA           NA
    Prevalence                  0.006289       0.012579            0            0
    Detection Rate              0.006289       0.009434            0            0
    Detection Prevalence        0.006289       0.009434            0            0
    Balanced Accuracy           1.000000       0.875000           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                 1.00000          1.000000            NA
    Specificity                 1.00000          1.000000             1
    Pos Pred Value              1.00000          1.000000            NA
    Neg Pred Value              1.00000          1.000000            NA
    Prevalence                  0.01258          0.003145             0
    Detection Rate              0.01258          0.003145             0
    Detection Prevalence        0.01258          0.003145             0
    Balanced Accuracy           1.00000          1.000000            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         1.00000
    Specificity                      1                     1         1.00000
    Pos Pred Value                  NA                    NA         1.00000
    Neg Pred Value                  NA                    NA         1.00000
    Prevalence                       0                     0         0.01258
    Detection Rate                   0                     0         0.01258
    Detection Prevalence             0                     0         0.01258
    Balanced Accuracy               NA                    NA         1.00000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA       1.00000        0.750000     0.750000
    Specificity                       1       0.99682        1.000000     1.000000
    Pos Pred Value                   NA       0.80000        1.000000     1.000000
    Neg Pred Value                   NA       1.00000        0.996825     0.996825
    Prevalence                        0       0.01258        0.012579     0.012579
    Detection Rate                    0       0.01258        0.009434     0.009434
    Detection Prevalence              0       0.01572        0.009434     0.009434
    Balanced Accuracy                NA       0.99841        0.875000     0.875000
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                 1.00000           NA       1.000000      1.00000
    Specificity                 0.99363            1       1.000000      1.00000
    Pos Pred Value              0.66667           NA       1.000000      1.00000
    Neg Pred Value              1.00000           NA       1.000000      1.00000
    Prevalence                  0.01258            0       0.006289      0.01258
    Detection Rate              0.01258            0       0.006289      0.01258
    Detection Prevalence        0.01887            0       0.006289      0.01258
    Balanced Accuracy           0.99682           NA       1.000000      1.00000
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                 1.00000           NA              NA      1.00000
    Specificity                 1.00000            1               1      1.00000
    Pos Pred Value              1.00000           NA              NA      1.00000
    Neg Pred Value              1.00000           NA              NA      1.00000
    Prevalence                  0.01258            0               0      0.01258
    Detection Rate              0.01258            0               0      0.01258
    Detection Prevalence        0.01258            0               0      0.01258
    Balanced Accuracy           1.00000           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity                1.00000              NA             NA      0.750000
    Specificity                1.00000               1              1      1.000000
    Pos Pred Value             1.00000              NA             NA      1.000000
    Neg Pred Value             1.00000              NA             NA      0.996825
    Prevalence                 0.01258               0              0      0.012579
    Detection Rate             0.01258               0              0      0.009434
    Detection Prevalence       0.01258               0              0      0.009434
    Balanced Accuracy          1.00000              NA             NA      0.875000
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity               1.00000              NA            NA            NA
    Specificity               1.00000               1             1             1
    Pos Pred Value            1.00000              NA            NA            NA
    Neg Pred Value            1.00000              NA            NA            NA
    Prevalence                0.01258               0             0             0
    Detection Rate            0.01258               0             0             0
    Detection Prevalence      0.01258               0             0             0
    Balanced Accuracy         1.00000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        1.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     0.996815
    Pos Pred Value             1.000000        1.000000           NaN     0.750000
    Neg Pred Value             1.000000        1.000000      0.996855     0.996815
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.006289        0.009434      0.000000     0.009434
    Detection Prevalence       0.006289        0.009434      0.000000     0.012579
    Balanced Accuracy          1.000000        1.000000      0.500000     0.873408
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       0.666667             NA      1.000000
    Specificity               1.00000       1.000000              1      0.993671
    Pos Pred Value            1.00000       1.000000             NA      0.500000
    Neg Pred Value            1.00000       0.996835             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.01258       0.006289              0      0.006289
    Detection Prevalence      0.01258       0.006289              0      0.012579
    Balanced Accuracy         1.00000       0.833333             NA      0.996835
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity               1.00000        1.00000           NA            NA
    Specificity               1.00000        1.00000            1             1
    Pos Pred Value            1.00000        1.00000           NA            NA
    Neg Pred Value            1.00000        1.00000           NA            NA
    Prevalence                0.01572        0.01572            0             0
    Detection Rate            0.01572        0.01572            0             0
    Detection Prevalence      0.01572        0.01572            0             0
    Balanced Accuracy         1.00000        1.00000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                 1.00000          1.000000      1.000000
    Specificity                 1.00000          1.000000      1.000000
    Pos Pred Value              1.00000          1.000000      1.000000
    Neg Pred Value              1.00000          1.000000      1.000000
    Prevalence                  0.01258          0.003145      0.003145
    Detection Rate              0.01258          0.003145      0.003145
    Detection Prevalence        0.01258          0.003145      0.003145
    Balanced Accuracy           1.00000          1.000000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9804            NA
    Specificity                1.00000              1      0.9825             1
    Pos Pred Value             1.00000             NA      0.9901            NA
    Neg Pred Value             1.00000             NA      0.9655            NA
    Prevalence                 0.01258              0      0.6415             0
    Detection Rate             0.01258              0      0.6289             0
    Detection Prevalence       0.01258              0      0.6352             0
    Balanced Accuracy          1.00000             NA      0.9814            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          1.000000                NA
    Specificity                   1.000000          1.000000          0.996855
    Pos Pred Value                     NaN          1.000000                NA
    Neg Pred Value                0.996855          1.000000                NA
    Prevalence                    0.003145          0.003145          0.000000
    Detection Rate                0.000000          0.003145          0.000000
    Detection Prevalence          0.000000          0.003145          0.003145
    Balanced Accuracy             0.500000          1.000000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000        1.00000
    Specificity                           1       1.00000        1.00000
    Pos Pred Value                       NA       1.00000        1.00000
    Neg Pred Value                       NA       1.00000        1.00000
    Prevalence                            0       0.01258        0.01258
    Detection Rate                        0       0.01258        0.01258
    Detection Prevalence                  0       0.01258        0.01258
    Balanced Accuracy                    NA       1.00000        1.00000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000           0.993711
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258           0.000000
    Detection Rate             0.01258           0.000000
    Detection Prevalence       0.01258           0.006289
    Balanced Accuracy          1.00000                 NA


**weighted random forest**


```R
prediction <-predict(wt.rf.500, newdata=test)
cm.wrf_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.wrf_mod
```


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              4        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        2       0     0     0
      AEDECOD            0        0        0     0        0       3     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       0     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               0        0        0     0        0       1     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              4          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        4
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 0          0      0      0              0        0
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        0     0       0     0       0     0
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      4        0     1       0     0       0     0
      AEHLGTCD             0      0        3     0       0     0       0     0
      AEHLT                0      0        0     3       0     0       0     0
      AEHLTCD              0      0        1     0       4     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       2     0
      AELLT                0      0        0     0       0     0       0     4
      AELLTCD              0      0        0     0       0     0       0     0
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       0     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      0        0     0       0     0       0     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      0
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      1
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                0     0        0     0      0        0       0      0
      AELLTCD              4     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      4        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               0     0        0     0      0        0       0      3
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 0     0        0     0      0        0       0      0
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      1     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              4        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       2        0      0     0
      AESDISAB           0        0      0      0       0        3      0     0
      AESDTH             0        0      0      0       0        0      0     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               0        0      0      0       0        0      0     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      0     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     0       0     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       0     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              4       0       0      0     0       0     0      0
      AESHOSP            0       2       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     5       0     0      0
      AESOCCD            0       0       0      0     0       5     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               0       0       0      0     0       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       1       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      0       0    0      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    0      0       0
      AEENDTC_TM           0          0      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    0      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    1      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    2      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              4          0      0      0       0    0      0       0
      AESTDTC_TM           0          1      0      0       0    0      0       0
      AESTRF               0          0      1      0       0    0      0       0
      AETERM               0          0      0      4       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 0          0      0      0       0  200      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          0          0          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          0          0          0         0
      QNAM_ESAM2             0         0          0          1          0         0
      QNAM_ESAM3             0         0          1          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      0       0      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      4       0      0
      STUDYID                 0          0           0      0       4      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                             
                   Accuracy : 0.9623         
                     95% CI : (0.935, 0.9804)
        No Information Rate : 0.6415         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.9359         
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity               1.00000              NA              NA           NA
    Specificity               0.99682               1               1            1
    Pos Pred Value            0.80000              NA              NA           NA
    Neg Pred Value            1.00000              NA              NA           NA
    Prevalence                0.01258               0               0            0
    Detection Rate            0.01258               0               0            0
    Detection Prevalence      0.01572               0               0            0
    Balanced Accuracy         0.99841              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000       0.750000           NA           NA
    Specificity                 1.000000       1.000000            1            1
    Pos Pred Value              1.000000       1.000000           NA           NA
    Neg Pred Value              1.000000       0.996825           NA           NA
    Prevalence                  0.006289       0.012579            0            0
    Detection Rate              0.006289       0.009434            0            0
    Detection Prevalence        0.006289       0.009434            0            0
    Balanced Accuracy           1.000000       0.875000           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                 1.00000          1.000000            NA
    Specificity                 1.00000          1.000000             1
    Pos Pred Value              1.00000          1.000000            NA
    Neg Pred Value              1.00000          1.000000            NA
    Prevalence                  0.01258          0.003145             0
    Detection Rate              0.01258          0.003145             0
    Detection Prevalence        0.01258          0.003145             0
    Balanced Accuracy           1.00000          1.000000            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         1.00000
    Specificity                      1              0.996855         1.00000
    Pos Pred Value                  NA                    NA         1.00000
    Neg Pred Value                  NA                    NA         1.00000
    Prevalence                       0              0.000000         0.01258
    Detection Rate                   0              0.000000         0.01258
    Detection Prevalence             0              0.003145         0.01258
    Balanced Accuracy               NA                    NA         1.00000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA       1.00000        0.750000     0.750000
    Specificity                       1       0.99682        1.000000     1.000000
    Pos Pred Value                   NA       0.80000        1.000000     1.000000
    Neg Pred Value                   NA       1.00000        0.996825     0.996825
    Prevalence                        0       0.01258        0.012579     0.012579
    Detection Rate                    0       0.01258        0.009434     0.009434
    Detection Prevalence              0       0.01572        0.009434     0.009434
    Balanced Accuracy                NA       0.99841        0.875000     0.875000
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                 1.00000           NA       1.000000      1.00000
    Specificity                 0.99363            1       1.000000      1.00000
    Pos Pred Value              0.66667           NA       1.000000      1.00000
    Neg Pred Value              1.00000           NA       1.000000      1.00000
    Prevalence                  0.01258            0       0.006289      0.01258
    Detection Rate              0.01258            0       0.006289      0.01258
    Detection Prevalence        0.01887            0       0.006289      0.01258
    Balanced Accuracy           0.99682           NA       1.000000      1.00000
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                 1.00000           NA              NA      1.00000
    Specificity                 1.00000            1               1      1.00000
    Pos Pred Value              1.00000           NA              NA      1.00000
    Neg Pred Value              1.00000           NA              NA      1.00000
    Prevalence                  0.01258            0               0      0.01258
    Detection Rate              0.01258            0               0      0.01258
    Detection Prevalence        0.01258            0               0      0.01258
    Balanced Accuracy           1.00000           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity                1.00000              NA             NA      0.750000
    Specificity                1.00000               1              1      1.000000
    Pos Pred Value             1.00000              NA             NA      1.000000
    Neg Pred Value             1.00000              NA             NA      0.996825
    Prevalence                 0.01258               0              0      0.012579
    Detection Rate             0.01258               0              0      0.009434
    Detection Prevalence       0.01258               0              0      0.009434
    Balanced Accuracy          1.00000              NA             NA      0.875000
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity               1.00000              NA            NA            NA
    Specificity               1.00000               1             1             1
    Pos Pred Value            1.00000              NA            NA            NA
    Neg Pred Value            1.00000              NA            NA            NA
    Prevalence                0.01258               0             0             0
    Detection Rate            0.01258               0             0             0
    Detection Prevalence      0.01258               0             0             0
    Balanced Accuracy         1.00000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        1.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     0.996815
    Pos Pred Value             1.000000        1.000000           NaN     0.750000
    Neg Pred Value             1.000000        1.000000      0.996855     0.996815
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.006289        0.009434      0.000000     0.009434
    Detection Prevalence       0.006289        0.009434      0.000000     0.012579
    Balanced Accuracy          1.000000        1.000000      0.500000     0.873408
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       0.666667             NA      1.000000
    Specificity               1.00000       1.000000              1      0.993671
    Pos Pred Value            1.00000       1.000000             NA      0.500000
    Neg Pred Value            1.00000       0.996835             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.01258       0.006289              0      0.006289
    Detection Prevalence      0.01258       0.006289              0      0.012579
    Balanced Accuracy         1.00000       0.833333             NA      0.996835
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity               1.00000        1.00000           NA            NA
    Specificity               1.00000        1.00000            1             1
    Pos Pred Value            1.00000        1.00000           NA            NA
    Neg Pred Value            1.00000        1.00000           NA            NA
    Prevalence                0.01572        0.01572            0             0
    Detection Rate            0.01572        0.01572            0             0
    Detection Prevalence      0.01572        0.01572            0             0
    Balanced Accuracy         1.00000        1.00000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                 1.00000          1.000000      1.000000
    Specificity                 1.00000          1.000000      1.000000
    Pos Pred Value              1.00000          1.000000      1.000000
    Neg Pred Value              1.00000          1.000000      1.000000
    Prevalence                  0.01258          0.003145      0.003145
    Detection Rate              0.01258          0.003145      0.003145
    Detection Prevalence        0.01258          0.003145      0.003145
    Balanced Accuracy           1.00000          1.000000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9804            NA
    Specificity                1.00000              1      0.9825             1
    Pos Pred Value             1.00000             NA      0.9901            NA
    Neg Pred Value             1.00000             NA      0.9655            NA
    Prevalence                 0.01258              0      0.6415             0
    Detection Rate             0.01258              0      0.6289             0
    Detection Prevalence       0.01258              0      0.6352             0
    Balanced Accuracy          1.00000             NA      0.9814            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          1.000000                NA
    Specificity                   1.000000          1.000000          0.996855
    Pos Pred Value                     NaN          1.000000                NA
    Neg Pred Value                0.996855          1.000000                NA
    Prevalence                    0.003145          0.003145          0.000000
    Detection Rate                0.000000          0.003145          0.000000
    Detection Prevalence          0.000000          0.003145          0.003145
    Balanced Accuracy             0.500000          1.000000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000        1.00000
    Specificity                           1       1.00000        1.00000
    Pos Pred Value                       NA       1.00000        1.00000
    Neg Pred Value                       NA       1.00000        1.00000
    Prevalence                            0       0.01258        0.01258
    Detection Rate                        0       0.01258        0.01258
    Detection Prevalence                  0       0.01258        0.01258
    Balanced Accuracy                    NA       1.00000        1.00000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000           0.996855
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258           0.000000
    Detection Rate             0.01258           0.000000
    Detection Prevalence       0.01258           0.003145
    Balanced Accuracy          1.00000                 NA


**feed forward nn**


```R
prediction <-predict(nnet_mod, newdata=test)
cm.nn_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.nn_mod
```

    Warning message in confusionMatrix.default(prediction, as.factor(response_test2)):
    "The data contains levels not found in the data, but they are empty and will be dropped."Warning message in levels(reference) != levels(data):
    "longer object length is not a multiple of shorter object length"Warning message in confusionMatrix.default(prediction, as.factor(response_test2)):
    "Levels are not in the same order for reference and data. Refactoring data to match."


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              4        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        2       0     0     0
      AEDECOD            0        0        0     0        0       4     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       0     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               0        0        0     0        0       0     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              4          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        4
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 0          0      0      0              0        0
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        0     0       0     0       0     0
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      4        0     1       0     0       0     0
      AEHLGTCD             0      0        3     0       0     0       0     0
      AEHLT                0      0        0     3       0     0       0     0
      AEHLTCD              0      0        1     0       4     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       2     0
      AELLT                0      0        0     0       0     0       0     4
      AELLTCD              0      0        0     0       0     0       0     0
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       0     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      0        0     0       0     0       0     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      0
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      0
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                0     0        0     0      0        0       0      0
      AELLTCD              4     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      4        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               0     0        0     0      0        0       0      4
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 0     0        0     0      0        0       0      0
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      0     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              4        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       2        0      0     0
      AESDISAB           0        0      0      0       0        2      0     0
      AESDTH             0        0      0      0       0        0      1     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               0        0      0      0       0        1      0     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      0     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     0       0     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       0     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              4       0       0      0     0       0     0      0
      AESHOSP            0       2       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     5       0     0      0
      AESOCCD            0       0       0      0     0       5     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               0       1       0      0     0       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       0       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      0       0    0      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    2      0       0
      AEENDTC_TM           0          0      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    0      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    0      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    2      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              4          0      0      0       0    0      0       0
      AESTDTC_TM           0          1      0      0       0    0      0       0
      AESTRF               0          0      1      0       0    0      0       0
      AETERM               0          0      0      4       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 0          0      0      0       0  199      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          0          0          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          1          0          0         0
      QNAM_ESAM2             0         0          0          1          0         0
      QNAM_ESAM3             0         0          0          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      0       0      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      4       1      0
      STUDYID                 0          0           0      0       3      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                             
                   Accuracy : 0.9654         
                     95% CI : (0.939, 0.9826)
        No Information Rate : 0.6415         
        P-Value [Acc > NIR] : < 2.2e-16      
                                             
                      Kappa : 0.9412         
     Mcnemar's Test P-Value : NA             
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity               1.00000              NA              NA           NA
    Specificity               0.99682               1               1            1
    Pos Pred Value            0.80000              NA              NA           NA
    Neg Pred Value            1.00000              NA              NA           NA
    Prevalence                0.01258               0               0            0
    Detection Rate            0.01258               0               0            0
    Detection Prevalence      0.01572               0               0            0
    Balanced Accuracy         0.99841              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 1.000000        1.00000           NA           NA
    Specificity                 1.000000        1.00000            1            1
    Pos Pred Value              1.000000        1.00000           NA           NA
    Neg Pred Value              1.000000        1.00000           NA           NA
    Prevalence                  0.006289        0.01258            0            0
    Detection Rate              0.006289        0.01258            0            0
    Detection Prevalence        0.006289        0.01258            0            0
    Balanced Accuracy           1.000000        1.00000           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                 1.00000          1.000000            NA
    Specificity                 0.99363          1.000000             1
    Pos Pred Value              0.66667          1.000000            NA
    Neg Pred Value              1.00000          1.000000            NA
    Prevalence                  0.01258          0.003145             0
    Detection Rate              0.01258          0.003145             0
    Detection Prevalence        0.01887          0.003145             0
    Balanced Accuracy           0.99682          1.000000            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         1.00000
    Specificity                      1                     1         1.00000
    Pos Pred Value                  NA                    NA         1.00000
    Neg Pred Value                  NA                    NA         1.00000
    Prevalence                       0                     0         0.01258
    Detection Rate                   0                     0         0.01258
    Detection Prevalence             0                     0         0.01258
    Balanced Accuracy               NA                    NA         1.00000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA       1.00000        0.750000     0.750000
    Specificity                       1       0.99682        1.000000     1.000000
    Pos Pred Value                   NA       0.80000        1.000000     1.000000
    Neg Pred Value                   NA       1.00000        0.996825     0.996825
    Prevalence                        0       0.01258        0.012579     0.012579
    Detection Rate                    0       0.01258        0.009434     0.009434
    Detection Prevalence              0       0.01572        0.009434     0.009434
    Balanced Accuracy                NA       0.99841        0.875000     0.875000
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                 1.00000           NA       1.000000      1.00000
    Specificity                 0.99682            1       1.000000      1.00000
    Pos Pred Value              0.80000           NA       1.000000      1.00000
    Neg Pred Value              1.00000           NA       1.000000      1.00000
    Prevalence                  0.01258            0       0.006289      0.01258
    Detection Rate              0.01258            0       0.006289      0.01258
    Detection Prevalence        0.01572            0       0.006289      0.01258
    Balanced Accuracy           0.99841           NA       1.000000      1.00000
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                 1.00000           NA              NA      1.00000
    Specificity                 1.00000            1               1      1.00000
    Pos Pred Value              1.00000           NA              NA      1.00000
    Neg Pred Value              1.00000           NA              NA      1.00000
    Prevalence                  0.01258            0               0      0.01258
    Detection Rate              0.01258            0               0      0.01258
    Detection Prevalence        0.01258            0               0      0.01258
    Balanced Accuracy           1.00000           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity                1.00000              NA             NA       1.00000
    Specificity                1.00000               1              1       1.00000
    Pos Pred Value             1.00000              NA             NA       1.00000
    Neg Pred Value             1.00000              NA             NA       1.00000
    Prevalence                 0.01258               0              0       0.01258
    Detection Rate             0.01258               0              0       0.01258
    Detection Prevalence       0.01258               0              0       0.01258
    Balanced Accuracy          1.00000              NA             NA       1.00000
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity               1.00000              NA            NA            NA
    Specificity               1.00000               1             1             1
    Pos Pred Value            1.00000              NA            NA            NA
    Neg Pred Value            1.00000              NA            NA            NA
    Prevalence                0.01258               0             0             0
    Detection Rate            0.01258               0             0             0
    Detection Prevalence      0.01258               0             0             0
    Balanced Accuracy         1.00000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                1.000000        0.666667      1.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     1.000000
    Pos Pred Value             1.000000        1.000000      1.000000     1.000000
    Neg Pred Value             1.000000        0.996835      1.000000     0.996825
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.006289        0.006289      0.003145     0.009434
    Detection Prevalence       0.006289        0.006289      0.003145     0.009434
    Balanced Accuracy          1.000000        0.833333      1.000000     0.875000
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               1.00000       0.666667             NA      1.000000
    Specificity               1.00000       1.000000              1      0.993671
    Pos Pred Value            1.00000       1.000000             NA      0.500000
    Neg Pred Value            1.00000       0.996835             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.01258       0.006289              0      0.006289
    Detection Prevalence      0.01258       0.006289              0      0.012579
    Balanced Accuracy         1.00000       0.833333             NA      0.996835
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity               1.00000        1.00000           NA            NA
    Specificity               1.00000        1.00000            1             1
    Pos Pred Value            1.00000        1.00000           NA            NA
    Neg Pred Value            1.00000        1.00000           NA            NA
    Prevalence                0.01572        0.01572            0             0
    Detection Rate            0.01572        0.01572            0             0
    Detection Prevalence      0.01572        0.01572            0             0
    Balanced Accuracy         1.00000        1.00000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                 1.00000          1.000000      1.000000
    Specificity                 1.00000          1.000000      1.000000
    Pos Pred Value              1.00000          1.000000      1.000000
    Neg Pred Value              1.00000          1.000000      1.000000
    Prevalence                  0.01258          0.003145      0.003145
    Detection Rate              0.01258          0.003145      0.003145
    Detection Prevalence        0.01258          0.003145      0.003145
    Balanced Accuracy           1.00000          1.000000      1.000000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9755            NA
    Specificity                1.00000              1      0.9737             1
    Pos Pred Value             1.00000             NA      0.9851            NA
    Neg Pred Value             1.00000             NA      0.9569            NA
    Prevalence                 0.01258              0      0.6415             0
    Detection Rate             0.01258              0      0.6258             0
    Detection Prevalence       0.01258              0      0.6352             0
    Balanced Accuracy          1.00000             NA      0.9746            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   1.000000          1.000000                NA
    Specificity                   1.000000          1.000000                 1
    Pos Pred Value                1.000000          1.000000                NA
    Neg Pred Value                1.000000          1.000000                NA
    Prevalence                    0.003145          0.003145                 0
    Detection Rate                0.003145          0.003145                 0
    Detection Prevalence          0.003145          0.003145                 0
    Balanced Accuracy             1.000000          1.000000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       1.00000       0.750000
    Specificity                           1       0.99682       1.000000
    Pos Pred Value                       NA       0.80000       1.000000
    Neg Pred Value                       NA       1.00000       0.996825
    Prevalence                            0       0.01258       0.012579
    Detection Rate                        0       0.01258       0.009434
    Detection Prevalence                  0       0.01572       0.009434
    Balanced Accuracy                    NA       0.99841       0.875000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000                  1
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258                  0
    Detection Rate             0.01258                  0
    Detection Prevalence       0.01258                  0
    Balanced Accuracy          1.00000                 NA


**xgboost**


```R
prediction <-predict(xgb_m1_1, newdata=test)
cm.x_mod <-confusionMatrix(prediction, as.factor(response_test2))
cm.x_mod
```


    Confusion Matrix and Statistics
    
                    Reference
    Prediction       AEACN AEACNDEV AEACNOTH AECAT AECONTRT AEDECOD AEDIR AEDUR
      AEACN              3        0        0     0        0       0     0     0
      AEACNDEV           0        0        0     0        0       0     0     0
      AEACNOTH           0        0        0     0        0       0     0     0
      AECAT              0        0        0     0        0       0     0     0
      AECONTRT           0        0        0     0        0       0     0     0
      AEDECOD            0        0        0     0        0       0     0     0
      AEDIR              0        0        0     0        0       0     0     0
      AEDUR              0        0        0     0        0       0     0     0
      AEENDTC            0        0        0     0        0       0     0     0
      AEENDTC_TM         0        0        0     0        0       0     0     0
      AEENDY             0        0        0     0        0       0     0     0
      AEENRF             0        0        0     0        0       0     0     0
      AEENRTP_AEENRF     0        0        0     0        0       0     0     0
      AEENRTPT           0        0        0     0        0       0     0     0
      AEENTPT            0        0        0     0        0       0     0     0
      AEHLGT             0        0        0     0        0       0     0     0
      AEHLGTCD           0        0        0     0        0       0     0     0
      AEHLT              0        0        0     0        0       0     0     0
      AEHLTCD            0        0        0     0        0       0     0     0
      AELAT              0        0        0     0        0       0     0     0
      AELIFTH            0        0        0     0        0       0     0     0
      AELLT              0        0        0     0        0       0     0     0
      AELLTCD            0        0        0     0        0       0     0     0
      AELOC              0        0        0     0        0       0     0     0
      AEMODIFY           0        0        0     0        0       0     0     0
      AEOUT              0        0        0     0        0       0     0     0
      AEPATT             0        0        0     0        0       0     0     0
      AEPORTOT           0        0        0     0        0       0     0     0
      AEPRESP            0        0        0     0        0       0     0     0
      AEPTCD             0        0        0     0        0       1     0     0
      AEREL              0        0        0     0        0       0     0     0
      AERELNST           0        0        0     0        0       0     0     0
      AESCAN             0        0        0     0        0       0     0     0
      AESCAT             0        0        0     0        0       0     0     0
      AESCONG            0        0        0     0        0       0     0     0
      AESDISAB           0        0        0     0        0       0     0     0
      AESDTH             0        0        0     0        0       0     0     0
      AESER              0        0        0     0        0       0     0     0
      AESEV              0        0        0     0        0       0     0     0
      AESHOSP            0        0        0     0        0       0     0     0
      AESLIFE            0        0        0     0        0       0     0     0
      AESMIE             0        0        0     0        0       0     0     0
      AESOC              0        0        0     0        0       0     0     0
      AESOCCD            0        0        0     0        0       0     0     0
      AESOD              0        0        0     0        0       0     0     0
      AESPID             0        0        0     0        0       0     0     0
      AESTDTC            0        0        0     0        0       0     0     0
      AESTDTC_TM         0        0        0     0        0       0     0     0
      AESTRF             0        0        0     0        0       0     0     0
      AETERM             0        0        0     0        0       0     0     0
      AETOXGR            0        0        0     0        0       0     0     0
      DROP               1        0        0     0        2       3     0     0
      DTHDTC             0        0        0     0        0       0     0     0
      FAORRES            0        0        0     0        0       0     0     0
      QNAM_AESI          0        0        0     0        0       0     0     0
      QNAM_COPD          0        0        0     0        0       0     0     0
      QNAM_ESAM1         0        0        0     0        0       0     0     0
      QNAM_ESAM2         0        0        0     0        0       0     0     0
      QNAM_ESAM3         0        0        0     0        0       0     0     0
      QNAM_EXER          0        0        0     0        0       0     0     0
      QNAM_EXLAB         0        0        0     0        0       0     0     0
      QNAM_EXSAB         0        0        0     0        0       0     0     0
      QNAM_EXSTER        0        0        0     0        0       0     0     0
      SITEID             0        0        0     0        0       0     0     0
      STUDYID            0        0        0     0        0       0     0     0
      SUBJID             0        0        0     0        0       0     0     0
      SUPPAE.QVAL        0        0        0     0        0       0     0     0
                    Reference
    Prediction       AEENDTC AEENDTC_TM AEENDY AEENRF AEENRTP_AEENRF AEENRTPT
      AEACN                0          0      0      0              0        0
      AEACNDEV             0          0      0      0              0        0
      AEACNOTH             0          0      0      0              0        0
      AECAT                0          0      0      0              0        0
      AECONTRT             0          0      0      0              0        0
      AEDECOD              0          0      0      0              0        0
      AEDIR                0          0      0      0              0        0
      AEDUR                0          0      0      0              0        0
      AEENDTC              2          0      0      0              0        0
      AEENDTC_TM           0          1      0      0              0        0
      AEENDY               0          0      0      0              0        0
      AEENRF               0          0      0      0              0        0
      AEENRTP_AEENRF       0          0      0      0              0        0
      AEENRTPT             0          0      0      0              0        0
      AEENTPT              0          0      0      0              0        0
      AEHLGT               0          0      0      0              0        0
      AEHLGTCD             0          0      0      0              0        0
      AEHLT                0          0      0      0              0        0
      AEHLTCD              0          0      0      0              0        0
      AELAT                0          0      0      0              0        0
      AELIFTH              0          0      0      0              0        0
      AELLT                0          0      0      0              0        0
      AELLTCD              0          0      0      0              0        0
      AELOC                0          0      0      0              0        0
      AEMODIFY             0          0      0      0              0        0
      AEOUT                0          0      0      0              0        0
      AEPATT               0          0      0      0              0        0
      AEPORTOT             0          0      0      0              0        0
      AEPRESP              0          0      0      0              0        0
      AEPTCD               0          0      0      0              0        0
      AEREL                0          0      0      0              0        0
      AERELNST             0          0      0      0              0        0
      AESCAN               0          0      0      0              0        0
      AESCAT               0          0      0      0              0        0
      AESCONG              0          0      0      0              0        0
      AESDISAB             0          0      0      0              0        0
      AESDTH               0          0      0      0              0        0
      AESER                0          0      0      0              0        0
      AESEV                0          0      0      0              0        0
      AESHOSP              0          0      0      0              0        0
      AESLIFE              0          0      0      0              0        0
      AESMIE               0          0      0      0              0        0
      AESOC                0          0      0      0              0        0
      AESOCCD              0          0      0      0              0        0
      AESOD                0          0      0      0              0        0
      AESPID               0          0      0      0              0        0
      AESTDTC              0          0      0      0              0        0
      AESTDTC_TM           0          0      0      0              0        0
      AESTRF               0          0      0      0              0        0
      AETERM               0          0      0      0              0        0
      AETOXGR              0          0      0      0              0        0
      DROP                 2          0      0      0              0        4
      DTHDTC               0          0      0      0              0        0
      FAORRES              0          0      0      0              0        0
      QNAM_AESI            0          0      0      0              0        0
      QNAM_COPD            0          0      0      0              0        0
      QNAM_ESAM1           0          0      0      0              0        0
      QNAM_ESAM2           0          0      0      0              0        0
      QNAM_ESAM3           0          0      0      0              0        0
      QNAM_EXER            0          0      0      0              0        0
      QNAM_EXLAB           0          0      0      0              0        0
      QNAM_EXSAB           0          0      0      0              0        0
      QNAM_EXSTER          0          0      0      0              0        0
      SITEID               0          0      0      0              0        0
      STUDYID              0          0      0      0              0        0
      SUBJID               0          0      0      0              0        0
      SUPPAE.QVAL          0          0      0      0              0        0
                    Reference
    Prediction       AEENTPT AEHLGT AEHLGTCD AEHLT AEHLTCD AELAT AELIFTH AELLT
      AEACN                0      0        0     0       0     0       0     0
      AEACNDEV             0      0        0     0       0     0       0     0
      AEACNOTH             0      0        0     0       0     0       0     0
      AECAT                0      0        0     0       0     0       0     0
      AECONTRT             0      0        0     0       0     0       0     0
      AEDECOD              0      0        1     0       0     0       0     1
      AEDIR                0      0        0     0       0     0       0     0
      AEDUR                0      0        0     0       0     0       0     0
      AEENDTC              0      0        0     0       0     0       0     0
      AEENDTC_TM           0      0        0     0       0     0       0     0
      AEENDY               0      0        0     0       0     0       0     0
      AEENRF               0      0        0     0       0     0       0     0
      AEENRTP_AEENRF       0      0        0     0       0     0       0     0
      AEENRTPT             0      0        0     0       0     0       0     0
      AEENTPT              0      0        0     0       0     0       0     0
      AEHLGT               0      1        1     0       0     0       0     0
      AEHLGTCD             0      0        0     0       0     0       0     0
      AEHLT                0      2        0     0       0     0       0     0
      AEHLTCD              0      0        2     2       1     0       0     0
      AELAT                0      0        0     0       0     0       0     0
      AELIFTH              0      0        0     0       0     0       0     0
      AELLT                0      0        0     2       1     0       0     1
      AELLTCD              0      0        0     0       0     0       0     2
      AELOC                0      0        0     0       0     0       0     0
      AEMODIFY             0      0        0     0       0     0       0     0
      AEOUT                0      0        0     0       0     0       0     0
      AEPATT               0      0        0     0       0     0       0     0
      AEPORTOT             0      0        0     0       0     0       0     0
      AEPRESP              0      0        0     0       0     0       0     0
      AEPTCD               0      0        0     0       0     0       0     0
      AEREL                0      0        0     0       0     0       0     0
      AERELNST             0      0        0     0       0     0       0     0
      AESCAN               0      0        0     0       0     0       0     0
      AESCAT               0      0        0     0       0     0       0     0
      AESCONG              0      0        0     0       0     0       0     0
      AESDISAB             0      0        0     0       0     0       0     0
      AESDTH               0      0        0     0       0     0       0     0
      AESER                0      0        0     0       0     0       0     0
      AESEV                0      0        0     0       0     0       0     0
      AESHOSP              0      0        0     0       0     0       0     0
      AESLIFE              0      0        0     0       0     0       0     0
      AESMIE               0      0        0     0       0     0       0     0
      AESOC                0      0        0     0       0     0       0     0
      AESOCCD              0      0        0     0       0     0       0     0
      AESOD                0      0        0     0       0     0       0     0
      AESPID               0      0        0     0       0     0       0     0
      AESTDTC              0      0        0     0       0     0       0     0
      AESTDTC_TM           0      0        0     0       0     0       0     0
      AESTRF               0      0        0     0       0     0       0     0
      AETERM               0      0        0     0       0     0       0     0
      AETOXGR              0      0        0     0       0     0       0     0
      DROP                 0      1        0     0       2     0       2     0
      DTHDTC               0      0        0     0       0     0       0     0
      FAORRES              0      0        0     0       0     0       0     0
      QNAM_AESI            0      0        0     0       0     0       0     0
      QNAM_COPD            0      0        0     0       0     0       0     0
      QNAM_ESAM1           0      0        0     0       0     0       0     0
      QNAM_ESAM2           0      0        0     0       0     0       0     0
      QNAM_ESAM3           0      0        0     0       0     0       0     0
      QNAM_EXER            0      0        0     0       0     0       0     0
      QNAM_EXLAB           0      0        0     0       0     0       0     0
      QNAM_EXSAB           0      0        0     0       0     0       0     0
      QNAM_EXSTER          0      0        0     0       0     0       0     0
      SITEID               0      0        0     0       0     0       0     0
      STUDYID              0      0        0     0       0     0       0     0
      SUBJID               0      0        0     0       0     0       0     0
      SUPPAE.QVAL          0      0        0     0       0     0       0     0
                    Reference
    Prediction       AELLTCD AELOC AEMODIFY AEOUT AEPATT AEPORTOT AEPRESP AEPTCD
      AEACN                0     0        0     0      0        0       0      0
      AEACNDEV             0     0        0     0      0        0       0      0
      AEACNOTH             0     0        0     0      0        0       0      0
      AECAT                0     0        0     0      0        0       0      0
      AECONTRT             0     0        0     0      0        0       0      0
      AEDECOD              0     0        0     0      0        0       0      1
      AEDIR                0     0        0     0      0        0       0      0
      AEDUR                0     0        0     0      0        0       0      0
      AEENDTC              0     0        0     0      0        0       0      0
      AEENDTC_TM           0     0        0     0      0        0       0      0
      AEENDY               0     0        0     0      0        0       0      0
      AEENRF               0     0        0     0      0        0       0      0
      AEENRTP_AEENRF       0     0        0     0      0        0       0      0
      AEENRTPT             0     0        0     0      0        0       0      0
      AEENTPT              0     0        0     0      0        0       0      0
      AEHLGT               0     0        0     0      0        0       0      0
      AEHLGTCD             0     0        0     0      0        0       0      0
      AEHLT                0     0        0     0      0        0       0      0
      AEHLTCD              0     0        0     0      0        0       0      0
      AELAT                0     0        0     0      0        0       0      0
      AELIFTH              0     0        0     0      0        0       0      0
      AELLT                0     0        0     0      0        0       0      0
      AELLTCD              2     0        0     0      0        0       0      0
      AELOC                0     0        0     0      0        0       0      0
      AEMODIFY             0     0        0     0      0        0       0      0
      AEOUT                0     0        0     4      0        0       0      0
      AEPATT               0     0        0     0      3        0       0      0
      AEPORTOT             0     0        0     0      0        0       0      0
      AEPRESP              0     0        0     0      0        0       0      0
      AEPTCD               0     0        0     0      0        0       0      2
      AEREL                0     0        0     0      0        0       0      0
      AERELNST             0     0        0     0      0        0       0      0
      AESCAN               0     0        0     0      0        0       0      0
      AESCAT               0     0        0     0      0        0       0      0
      AESCONG              0     0        0     0      0        0       0      0
      AESDISAB             0     0        0     0      0        0       0      0
      AESDTH               0     0        0     0      0        0       0      0
      AESER                0     0        0     0      0        0       0      0
      AESEV                0     0        0     0      0        0       0      0
      AESHOSP              0     0        0     0      0        0       0      0
      AESLIFE              0     0        0     0      0        0       0      0
      AESMIE               0     0        0     0      0        0       0      0
      AESOC                0     0        0     0      0        0       0      0
      AESOCCD              0     0        0     0      0        0       0      0
      AESOD                0     0        0     0      0        0       0      0
      AESPID               0     0        0     0      0        0       0      0
      AESTDTC              0     0        0     0      0        0       0      0
      AESTDTC_TM           0     0        0     0      0        0       0      0
      AESTRF               0     0        0     0      0        0       0      0
      AETERM               0     0        0     0      0        0       0      0
      AETOXGR              0     0        0     0      0        0       0      0
      DROP                 2     0        0     0      1        0       0      1
      DTHDTC               0     0        0     0      0        0       0      0
      FAORRES              0     0        0     0      0        0       0      0
      QNAM_AESI            0     0        0     0      0        0       0      0
      QNAM_COPD            0     0        0     0      0        0       0      0
      QNAM_ESAM1           0     0        0     0      0        0       0      0
      QNAM_ESAM2           0     0        0     0      0        0       0      0
      QNAM_ESAM3           0     0        0     0      0        0       0      0
      QNAM_EXER            0     0        0     0      0        0       0      0
      QNAM_EXLAB           0     0        0     0      0        0       0      0
      QNAM_EXSAB           0     0        0     0      0        0       0      0
      QNAM_EXSTER          0     0        0     0      0        0       0      0
      SITEID               0     0        0     0      0        0       0      0
      STUDYID              0     0        0     0      0        0       0      0
      SUBJID               0     0        0     0      0        0       0      0
      SUPPAE.QVAL          0     0        0     0      0        0       0      0
                    Reference
    Prediction       AEREL AERELNST AESCAN AESCAT AESCONG AESDISAB AESDTH AESER
      AEACN              0        0      0      0       0        0      0     0
      AEACNDEV           0        0      0      0       0        0      0     0
      AEACNOTH           0        0      0      0       0        0      0     0
      AECAT              0        0      0      0       0        0      0     0
      AECONTRT           0        0      0      0       0        0      0     0
      AEDECOD            0        0      0      0       0        0      0     0
      AEDIR              0        0      0      0       0        0      0     0
      AEDUR              0        0      0      0       0        0      0     0
      AEENDTC            0        0      0      0       0        0      0     0
      AEENDTC_TM         0        0      0      0       0        0      0     0
      AEENDY             0        0      0      0       0        0      0     0
      AEENRF             0        0      0      0       0        0      0     0
      AEENRTP_AEENRF     0        0      0      0       0        0      0     0
      AEENRTPT           0        0      0      0       0        0      0     0
      AEENTPT            0        0      0      0       0        0      0     0
      AEHLGT             0        0      0      0       0        0      0     0
      AEHLGTCD           0        0      0      0       0        0      0     0
      AEHLT              0        0      0      0       0        0      0     0
      AEHLTCD            0        0      0      0       0        0      0     0
      AELAT              0        0      0      0       0        0      0     0
      AELIFTH            0        0      0      0       0        0      0     0
      AELLT              0        0      0      0       0        0      0     0
      AELLTCD            0        0      0      0       0        0      0     0
      AELOC              0        0      0      0       0        0      0     0
      AEMODIFY           0        0      0      0       0        0      0     0
      AEOUT              0        0      0      0       0        0      0     0
      AEPATT             0        0      0      0       0        0      0     0
      AEPORTOT           0        0      0      0       0        0      0     0
      AEPRESP            0        0      0      0       0        0      0     0
      AEPTCD             0        0      0      0       0        0      0     0
      AEREL              0        0      0      0       0        0      0     0
      AERELNST           0        0      0      0       0        0      0     0
      AESCAN             0        0      0      0       0        0      0     0
      AESCAT             0        0      0      0       0        0      0     0
      AESCONG            0        0      0      0       0        0      0     0
      AESDISAB           0        0      0      0       0        0      0     0
      AESDTH             0        0      0      0       0        0      0     0
      AESER              0        0      0      0       0        0      0     3
      AESEV              0        0      0      0       0        0      0     0
      AESHOSP            0        0      0      0       0        0      0     0
      AESLIFE            0        0      0      0       0        0      0     0
      AESMIE             0        0      0      0       0        0      0     0
      AESOC              0        0      0      0       0        0      0     0
      AESOCCD            0        0      0      0       0        0      0     0
      AESOD              0        0      0      0       0        0      0     0
      AESPID             0        0      0      0       0        0      0     0
      AESTDTC            0        0      0      0       0        0      0     0
      AESTDTC_TM         0        0      0      0       0        0      0     0
      AESTRF             0        0      0      0       0        0      0     0
      AETERM             0        0      0      0       0        0      0     0
      AETOXGR            0        0      0      0       0        0      0     0
      DROP               4        0      0      0       2        3      1     1
      DTHDTC             0        0      0      0       0        0      0     0
      FAORRES            0        0      0      0       0        0      0     0
      QNAM_AESI          0        0      0      0       0        0      0     0
      QNAM_COPD          0        0      0      0       0        0      0     0
      QNAM_ESAM1         0        0      0      0       0        0      0     0
      QNAM_ESAM2         0        0      0      0       0        0      0     0
      QNAM_ESAM3         0        0      0      0       0        0      0     0
      QNAM_EXER          0        0      0      0       0        0      0     0
      QNAM_EXLAB         0        0      0      0       0        0      0     0
      QNAM_EXSAB         0        0      0      0       0        0      0     0
      QNAM_EXSTER        0        0      0      0       0        0      0     0
      SITEID             0        0      0      0       0        0      0     0
      STUDYID            0        0      0      0       0        0      0     0
      SUBJID             0        0      0      0       0        0      0     0
      SUPPAE.QVAL        0        0      0      0       0        0      0     0
                    Reference
    Prediction       AESEV AESHOSP AESLIFE AESMIE AESOC AESOCCD AESOD AESPID
      AEACN              0       0       0      0     0       0     0      0
      AEACNDEV           0       0       0      0     0       0     0      0
      AEACNOTH           0       0       0      0     0       0     0      0
      AECAT              0       0       0      0     0       0     0      0
      AECONTRT           0       0       0      0     0       0     0      0
      AEDECOD            0       0       0      0     0       1     0      0
      AEDIR              0       0       0      0     0       0     0      0
      AEDUR              0       0       0      0     0       0     0      0
      AEENDTC            0       0       0      0     0       0     0      0
      AEENDTC_TM         0       0       0      0     0       0     0      0
      AEENDY             0       0       0      0     0       0     0      0
      AEENRF             0       0       0      0     0       0     0      0
      AEENRTP_AEENRF     0       0       0      0     0       0     0      0
      AEENRTPT           0       0       0      0     0       0     0      0
      AEENTPT            0       0       0      0     0       0     0      0
      AEHLGT             0       0       0      0     0       0     0      0
      AEHLGTCD           0       0       0      0     0       0     0      0
      AEHLT              0       0       0      0     0       0     0      0
      AEHLTCD            0       0       0      0     0       0     0      0
      AELAT              0       0       0      0     0       0     0      0
      AELIFTH            0       0       0      0     0       0     0      0
      AELLT              0       0       0      0     0       0     0      0
      AELLTCD            0       0       0      0     0       0     0      0
      AELOC              0       0       0      0     0       0     0      0
      AEMODIFY           0       0       0      0     0       0     0      0
      AEOUT              0       0       0      0     0       0     0      0
      AEPATT             0       0       0      0     0       0     0      0
      AEPORTOT           0       0       0      0     0       0     0      0
      AEPRESP            0       0       0      0     0       0     0      0
      AEPTCD             0       0       0      0     0       0     0      0
      AEREL              0       0       0      0     0       0     0      0
      AERELNST           0       0       0      0     0       0     0      0
      AESCAN             0       0       0      0     0       0     0      0
      AESCAT             0       0       0      0     0       0     0      0
      AESCONG            0       0       0      0     0       0     0      0
      AESDISAB           0       0       0      0     0       0     0      0
      AESDTH             0       0       0      0     0       0     0      0
      AESER              0       0       0      0     0       0     0      0
      AESEV              0       0       0      0     0       0     0      0
      AESHOSP            0       0       0      0     0       0     0      0
      AESLIFE            0       0       0      0     0       0     0      0
      AESMIE             0       0       0      2     0       0     0      0
      AESOC              0       0       0      0     2       0     0      0
      AESOCCD            0       0       0      0     0       4     0      0
      AESOD              0       0       0      0     0       0     0      0
      AESPID             0       0       0      0     0       0     0      0
      AESTDTC            0       0       0      0     0       0     0      0
      AESTDTC_TM         0       0       0      0     0       0     0      0
      AESTRF             0       0       0      0     0       0     0      0
      AETERM             0       0       0      0     0       0     0      0
      AETOXGR            0       0       0      0     0       0     0      0
      DROP               4       3       0      0     3       0     0      0
      DTHDTC             0       0       0      0     0       0     0      0
      FAORRES            0       0       0      0     0       0     0      0
      QNAM_AESI          0       0       0      0     0       0     0      0
      QNAM_COPD          0       0       0      0     0       0     0      0
      QNAM_ESAM1         0       0       0      0     0       0     0      0
      QNAM_ESAM2         0       0       0      0     0       0     0      0
      QNAM_ESAM3         0       0       0      0     0       0     0      0
      QNAM_EXER          0       0       0      0     0       0     0      0
      QNAM_EXLAB         0       0       0      0     0       0     0      0
      QNAM_EXSAB         0       0       0      0     0       0     0      0
      QNAM_EXSTER        0       0       0      0     0       0     0      0
      SITEID             0       0       0      0     0       0     0      0
      STUDYID            0       0       0      0     0       0     0      0
      SUBJID             0       0       0      0     0       0     0      0
      SUPPAE.QVAL        0       0       0      0     0       0     0      0
                    Reference
    Prediction       AESTDTC AESTDTC_TM AESTRF AETERM AETOXGR DROP DTHDTC FAORRES
      AEACN                0          0      0      0       0    1      0       0
      AEACNDEV             0          0      0      0       0    0      0       0
      AEACNOTH             0          0      0      0       0    0      0       0
      AECAT                0          0      0      0       0    0      0       0
      AECONTRT             0          0      0      0       0    0      0       0
      AEDECOD              0          0      0      0       0    0      0       0
      AEDIR                0          0      0      0       0    0      0       0
      AEDUR                0          0      0      0       0    0      0       0
      AEENDTC              0          0      0      0       0    0      0       0
      AEENDTC_TM           0          0      0      0       0    0      0       0
      AEENDY               0          0      0      0       0    0      0       0
      AEENRF               0          0      0      0       0    0      0       0
      AEENRTP_AEENRF       0          0      0      0       0    0      0       0
      AEENRTPT             0          0      0      0       0    0      0       0
      AEENTPT              0          0      0      0       0    0      0       0
      AEHLGT               0          0      0      0       0    0      0       0
      AEHLGTCD             0          0      0      0       0    0      0       0
      AEHLT                0          0      0      0       0    0      0       0
      AEHLTCD              0          0      0      0       0    0      0       0
      AELAT                0          0      0      0       0    0      0       0
      AELIFTH              0          0      0      0       0    0      0       0
      AELLT                0          0      0      0       0    0      0       0
      AELLTCD              0          0      0      0       0    0      0       0
      AELOC                0          0      0      0       0    0      0       0
      AEMODIFY             0          0      0      0       0    0      0       0
      AEOUT                0          0      0      0       0    0      0       0
      AEPATT               0          0      0      0       0    0      0       0
      AEPORTOT             0          0      0      0       0    0      0       0
      AEPRESP              0          0      0      0       0    0      0       0
      AEPTCD               0          0      0      0       0    0      0       0
      AEREL                0          0      0      0       0    0      0       0
      AERELNST             0          0      0      0       0    0      0       0
      AESCAN               0          0      0      0       0    0      0       0
      AESCAT               0          0      0      0       0    0      0       0
      AESCONG              0          0      0      0       0    0      0       0
      AESDISAB             0          0      0      0       0    0      0       0
      AESDTH               0          0      0      0       0    0      0       0
      AESER                0          0      0      0       0    0      0       0
      AESEV                0          0      0      0       0    0      0       0
      AESHOSP              0          0      0      0       0    0      0       0
      AESLIFE              0          0      0      0       0    0      0       0
      AESMIE               0          0      0      0       0    0      0       0
      AESOC                0          0      0      0       0    0      0       0
      AESOCCD              0          0      0      0       0    0      0       0
      AESOD                0          0      0      0       0    0      0       0
      AESPID               0          0      0      0       0    0      0       0
      AESTDTC              3          0      0      0       0    0      0       0
      AESTDTC_TM           0          0      0      0       0    0      0       0
      AESTRF               0          0      0      0       0    0      0       0
      AETERM               0          0      0      4       0    0      0       0
      AETOXGR              0          0      0      0       0    0      0       0
      DROP                 1          1      1      0       0  203      0       0
      DTHDTC               0          0      0      0       0    0      0       0
      FAORRES              0          0      0      0       0    0      0       0
      QNAM_AESI            0          0      0      0       0    0      0       0
      QNAM_COPD            0          0      0      0       0    0      0       0
      QNAM_ESAM1           0          0      0      0       0    0      0       0
      QNAM_ESAM2           0          0      0      0       0    0      0       0
      QNAM_ESAM3           0          0      0      0       0    0      0       0
      QNAM_EXER            0          0      0      0       0    0      0       0
      QNAM_EXLAB           0          0      0      0       0    0      0       0
      QNAM_EXSAB           0          0      0      0       0    0      0       0
      QNAM_EXSTER          0          0      0      0       0    0      0       0
      SITEID               0          0      0      0       0    0      0       0
      STUDYID              0          0      0      0       0    0      0       0
      SUBJID               0          0      0      0       0    0      0       0
      SUPPAE.QVAL          0          0      0      0       0    0      0       0
                    Reference
    Prediction       QNAM_AESI QNAM_COPD QNAM_ESAM1 QNAM_ESAM2 QNAM_ESAM3 QNAM_EXER
      AEACN                  0         0          0          0          0         0
      AEACNDEV               0         0          0          0          0         0
      AEACNOTH               0         0          0          0          0         0
      AECAT                  0         0          0          0          0         0
      AECONTRT               0         0          0          0          0         0
      AEDECOD                0         0          0          0          0         0
      AEDIR                  0         0          0          0          0         0
      AEDUR                  0         0          0          0          0         0
      AEENDTC                0         0          0          0          0         0
      AEENDTC_TM             0         0          0          0          0         0
      AEENDY                 0         0          0          0          0         0
      AEENRF                 0         0          0          0          0         0
      AEENRTP_AEENRF         0         0          0          0          0         0
      AEENRTPT               0         0          0          0          0         0
      AEENTPT                0         0          0          0          0         0
      AEHLGT                 0         0          0          0          0         0
      AEHLGTCD               0         0          0          0          0         0
      AEHLT                  0         0          0          0          0         0
      AEHLTCD                0         0          0          0          0         0
      AELAT                  0         0          0          0          0         0
      AELIFTH                0         0          0          0          0         0
      AELLT                  0         0          0          0          0         0
      AELLTCD                0         0          0          0          0         0
      AELOC                  0         0          0          0          0         0
      AEMODIFY               0         0          0          0          0         0
      AEOUT                  0         0          0          0          0         0
      AEPATT                 0         0          0          0          0         0
      AEPORTOT               0         0          0          0          0         0
      AEPRESP                0         0          0          0          0         0
      AEPTCD                 0         0          0          0          0         0
      AEREL                  0         0          0          0          0         0
      AERELNST               0         0          0          0          0         0
      AESCAN                 0         0          0          0          0         0
      AESCAT                 0         0          0          0          0         0
      AESCONG                0         0          0          0          0         0
      AESDISAB               0         0          0          0          0         0
      AESDTH                 0         0          0          0          0         0
      AESER                  0         0          0          0          0         0
      AESEV                  0         0          0          0          0         0
      AESHOSP                0         0          0          0          0         0
      AESLIFE                0         0          0          0          0         0
      AESMIE                 0         0          0          0          0         0
      AESOC                  0         0          0          0          0         0
      AESOCCD                0         0          0          0          0         0
      AESOD                  0         0          0          0          0         0
      AESPID                 0         0          0          0          0         0
      AESTDTC                0         0          0          0          0         0
      AESTDTC_TM             0         0          0          0          0         0
      AESTRF                 0         0          0          0          0         0
      AETERM                 0         0          0          0          0         0
      AETOXGR                0         0          0          0          0         0
      DROP                   0         0          1          1          0         0
      DTHDTC                 0         0          0          0          0         0
      FAORRES                0         0          0          0          0         0
      QNAM_AESI              0         0          0          0          0         0
      QNAM_COPD              0         0          0          0          0         0
      QNAM_ESAM1             0         0          0          0          0         0
      QNAM_ESAM2             0         0          0          0          0         0
      QNAM_ESAM3             0         0          0          0          0         0
      QNAM_EXER              0         0          0          0          0         0
      QNAM_EXLAB             0         0          0          0          0         0
      QNAM_EXSAB             0         0          0          0          0         0
      QNAM_EXSTER            0         0          0          0          0         0
      SITEID                 0         0          0          0          0         0
      STUDYID                0         0          0          0          0         0
      SUBJID                 0         0          0          0          0         0
      SUPPAE.QVAL            0         0          0          0          0         0
                    Reference
    Prediction       QNAM_EXLAB QNAM_EXSAB QNAM_EXSTER SITEID STUDYID SUBJID
      AEACN                   0          0           0      0       0      0
      AEACNDEV                0          0           0      0       0      0
      AEACNOTH                0          0           0      0       0      0
      AECAT                   0          0           0      0       0      0
      AECONTRT                0          0           0      0       0      0
      AEDECOD                 0          0           0      0       0      0
      AEDIR                   0          0           0      0       0      0
      AEDUR                   0          0           0      0       0      0
      AEENDTC                 0          0           0      0       0      0
      AEENDTC_TM              0          0           0      0       0      0
      AEENDY                  0          0           0      0       0      0
      AEENRF                  0          0           0      0       0      0
      AEENRTP_AEENRF          0          0           0      0       0      0
      AEENRTPT                0          0           0      0       0      0
      AEENTPT                 0          0           0      0       0      0
      AEHLGT                  0          0           0      0       0      0
      AEHLGTCD                0          0           0      0       0      0
      AEHLT                   0          0           0      0       0      0
      AEHLTCD                 0          0           0      0       0      0
      AELAT                   0          0           0      0       0      0
      AELIFTH                 0          0           0      0       0      0
      AELLT                   0          0           0      0       0      0
      AELLTCD                 0          0           0      0       0      0
      AELOC                   0          0           0      0       0      0
      AEMODIFY                0          0           0      0       0      0
      AEOUT                   0          0           0      0       0      0
      AEPATT                  0          0           0      0       0      0
      AEPORTOT                0          0           0      0       0      0
      AEPRESP                 0          0           0      0       0      0
      AEPTCD                  0          0           0      0       0      0
      AEREL                   0          0           0      0       0      0
      AERELNST                0          0           0      0       0      0
      AESCAN                  0          0           0      0       0      0
      AESCAT                  0          0           0      0       0      0
      AESCONG                 0          0           0      0       0      0
      AESDISAB                0          0           0      0       0      0
      AESDTH                  0          0           0      0       0      0
      AESER                   0          0           0      0       0      0
      AESEV                   0          0           0      0       0      0
      AESHOSP                 0          0           0      0       0      0
      AESLIFE                 0          0           0      0       0      0
      AESMIE                  0          0           0      0       0      0
      AESOC                   0          0           0      0       0      0
      AESOCCD                 0          0           0      0       0      0
      AESOD                   0          0           0      0       0      0
      AESPID                  0          0           0      0       0      0
      AESTDTC                 0          0           0      0       0      0
      AESTDTC_TM              0          0           0      0       0      0
      AESTRF                  0          0           0      0       0      0
      AETERM                  0          0           0      0       0      0
      AETOXGR                 0          0           0      0       0      0
      DROP                    0          0           0      4       4      0
      DTHDTC                  0          0           0      0       0      0
      FAORRES                 0          0           0      0       0      0
      QNAM_AESI               0          0           0      0       0      0
      QNAM_COPD               0          0           0      0       0      0
      QNAM_ESAM1              0          0           0      0       0      0
      QNAM_ESAM2              0          0           0      0       0      0
      QNAM_ESAM3              0          0           0      0       0      0
      QNAM_EXER               0          0           0      0       0      0
      QNAM_EXLAB              0          0           0      0       0      0
      QNAM_EXSAB              0          0           0      0       0      0
      QNAM_EXSTER             0          0           0      0       0      0
      SITEID                  0          0           0      0       0      0
      STUDYID                 0          0           0      0       0      0
      SUBJID                  0          0           0      0       0      4
      SUPPAE.QVAL             0          0           0      0       0      0
                    Reference
    Prediction       SUPPAE.QVAL
      AEACN                    0
      AEACNDEV                 0
      AEACNOTH                 0
      AECAT                    0
      AECONTRT                 0
      AEDECOD                  0
      AEDIR                    0
      AEDUR                    0
      AEENDTC                  0
      AEENDTC_TM               0
      AEENDY                   0
      AEENRF                   0
      AEENRTP_AEENRF           0
      AEENRTPT                 0
      AEENTPT                  0
      AEHLGT                   0
      AEHLGTCD                 0
      AEHLT                    0
      AEHLTCD                  0
      AELAT                    0
      AELIFTH                  0
      AELLT                    0
      AELLTCD                  0
      AELOC                    0
      AEMODIFY                 0
      AEOUT                    0
      AEPATT                   0
      AEPORTOT                 0
      AEPRESP                  0
      AEPTCD                   0
      AEREL                    0
      AERELNST                 0
      AESCAN                   0
      AESCAT                   0
      AESCONG                  0
      AESDISAB                 0
      AESDTH                   0
      AESER                    0
      AESEV                    0
      AESHOSP                  0
      AESLIFE                  0
      AESMIE                   0
      AESOC                    0
      AESOCCD                  0
      AESOD                    0
      AESPID                   0
      AESTDTC                  0
      AESTDTC_TM               0
      AESTRF                   0
      AETERM                   0
      AETOXGR                  0
      DROP                     0
      DTHDTC                   0
      FAORRES                  0
      QNAM_AESI                0
      QNAM_COPD                0
      QNAM_ESAM1               0
      QNAM_ESAM2               0
      QNAM_ESAM3               0
      QNAM_EXER                0
      QNAM_EXLAB               0
      QNAM_EXSAB               0
      QNAM_EXSTER              0
      SITEID                   0
      STUDYID                  0
      SUBJID                   0
      SUPPAE.QVAL              0
    
    Overall Statistics
                                              
                   Accuracy : 0.7704          
                     95% CI : (0.7202, 0.8155)
        No Information Rate : 0.6415          
        P-Value [Acc > NIR] : 4.902e-07       
                                              
                      Kappa : 0.5189          
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: AEACN Class: AEACNDEV Class: AEACNOTH Class: AECAT
    Sensitivity              0.750000              NA              NA           NA
    Specificity              0.996815               1               1            1
    Pos Pred Value           0.750000              NA              NA           NA
    Neg Pred Value           0.996815              NA              NA           NA
    Prevalence               0.012579               0               0            0
    Detection Rate           0.009434               0               0            0
    Detection Prevalence     0.012579               0               0            0
    Balanced Accuracy        0.873408              NA              NA           NA
                         Class: AECONTRT Class: AEDECOD Class: AEDIR Class: AEDUR
    Sensitivity                 0.000000        0.00000           NA           NA
    Specificity                 1.000000        0.98726            1            1
    Pos Pred Value                   NaN        0.00000           NA           NA
    Neg Pred Value              0.993711        0.98726           NA           NA
    Prevalence                  0.006289        0.01258            0            0
    Detection Rate              0.000000        0.00000            0            0
    Detection Prevalence        0.000000        0.01258            0            0
    Balanced Accuracy           0.500000        0.49363           NA           NA
                         Class: AEENDTC Class: AEENDTC_TM Class: AEENDY
    Sensitivity                0.500000          1.000000            NA
    Specificity                1.000000          1.000000             1
    Pos Pred Value             1.000000          1.000000            NA
    Neg Pred Value             0.993671          1.000000            NA
    Prevalence                 0.012579          0.003145             0
    Detection Rate             0.006289          0.003145             0
    Detection Prevalence       0.006289          0.003145             0
    Balanced Accuracy          0.750000          1.000000            NA
                         Class: AEENRF Class: AEENRTP_AEENRF Class: AEENRTPT
    Sensitivity                     NA                    NA         0.00000
    Specificity                      1                     1         1.00000
    Pos Pred Value                  NA                    NA             NaN
    Neg Pred Value                  NA                    NA         0.98742
    Prevalence                       0                     0         0.01258
    Detection Rate                   0                     0         0.00000
    Detection Prevalence             0                     0         0.00000
    Balanced Accuracy               NA                    NA         0.50000
                         Class: AEENTPT Class: AEHLGT Class: AEHLGTCD Class: AEHLT
    Sensitivity                      NA      0.250000         0.00000     0.000000
    Specificity                       1      0.996815         1.00000     0.993631
    Pos Pred Value                   NA      0.500000             NaN     0.000000
    Neg Pred Value                   NA      0.990506         0.98742     0.987342
    Prevalence                        0      0.012579         0.01258     0.012579
    Detection Rate                    0      0.003145         0.00000     0.000000
    Detection Prevalence              0      0.006289         0.00000     0.006289
    Balanced Accuracy                NA      0.623408         0.50000     0.496815
                         Class: AEHLTCD Class: AELAT Class: AELIFTH Class: AELLT
    Sensitivity                0.250000           NA       0.000000     0.250000
    Specificity                0.987261            1       1.000000     0.990446
    Pos Pred Value             0.200000           NA            NaN     0.250000
    Neg Pred Value             0.990415           NA       0.993711     0.990446
    Prevalence                 0.012579            0       0.006289     0.012579
    Detection Rate             0.003145            0       0.000000     0.003145
    Detection Prevalence       0.015723            0       0.000000     0.012579
    Balanced Accuracy          0.618631           NA       0.500000     0.620223
                         Class: AELLTCD Class: AELOC Class: AEMODIFY Class: AEOUT
    Sensitivity                0.500000           NA              NA      1.00000
    Specificity                0.993631            1               1      1.00000
    Pos Pred Value             0.500000           NA              NA      1.00000
    Neg Pred Value             0.993631           NA              NA      1.00000
    Prevalence                 0.012579            0               0      0.01258
    Detection Rate             0.006289            0               0      0.01258
    Detection Prevalence       0.012579            0               0      0.01258
    Balanced Accuracy          0.746815           NA              NA      1.00000
                         Class: AEPATT Class: AEPORTOT Class: AEPRESP Class: AEPTCD
    Sensitivity               0.750000              NA             NA      0.500000
    Specificity               1.000000               1              1      0.996815
    Pos Pred Value            1.000000              NA             NA      0.666667
    Neg Pred Value            0.996825              NA             NA      0.993651
    Prevalence                0.012579               0              0      0.012579
    Detection Rate            0.009434               0              0      0.006289
    Detection Prevalence      0.009434               0              0      0.009434
    Balanced Accuracy         0.875000              NA             NA      0.748408
                         Class: AEREL Class: AERELNST Class: AESCAN Class: AESCAT
    Sensitivity               0.00000              NA            NA            NA
    Specificity               1.00000               1             1             1
    Pos Pred Value                NaN              NA            NA            NA
    Neg Pred Value            0.98742              NA            NA            NA
    Prevalence                0.01258               0             0             0
    Detection Rate            0.00000               0             0             0
    Detection Prevalence      0.00000               0             0             0
    Balanced Accuracy         0.50000              NA            NA            NA
                         Class: AESCONG Class: AESDISAB Class: AESDTH Class: AESER
    Sensitivity                0.000000        0.000000      0.000000     0.750000
    Specificity                1.000000        1.000000      1.000000     1.000000
    Pos Pred Value                  NaN             NaN           NaN     1.000000
    Neg Pred Value             0.993711        0.990566      0.996855     0.996825
    Prevalence                 0.006289        0.009434      0.003145     0.012579
    Detection Rate             0.000000        0.000000      0.000000     0.009434
    Detection Prevalence       0.000000        0.000000      0.000000     0.009434
    Balanced Accuracy          0.500000        0.500000      0.500000     0.875000
                         Class: AESEV Class: AESHOSP Class: AESLIFE Class: AESMIE
    Sensitivity               0.00000       0.000000             NA      1.000000
    Specificity               1.00000       1.000000              1      1.000000
    Pos Pred Value                NaN            NaN             NA      1.000000
    Neg Pred Value            0.98742       0.990566             NA      1.000000
    Prevalence                0.01258       0.009434              0      0.006289
    Detection Rate            0.00000       0.000000              0      0.006289
    Detection Prevalence      0.00000       0.000000              0      0.006289
    Balanced Accuracy         0.50000       0.500000             NA      1.000000
                         Class: AESOC Class: AESOCCD Class: AESOD Class: AESPID
    Sensitivity              0.400000        0.80000           NA            NA
    Specificity              1.000000        1.00000            1             1
    Pos Pred Value           1.000000        1.00000           NA            NA
    Neg Pred Value           0.990506        0.99682           NA            NA
    Prevalence               0.015723        0.01572            0             0
    Detection Rate           0.006289        0.01258            0             0
    Detection Prevalence     0.006289        0.01258            0             0
    Balanced Accuracy        0.700000        0.90000           NA            NA
                         Class: AESTDTC Class: AESTDTC_TM Class: AESTRF
    Sensitivity                0.750000          0.000000      0.000000
    Specificity                1.000000          1.000000      1.000000
    Pos Pred Value             1.000000               NaN           NaN
    Neg Pred Value             0.996825          0.996855      0.996855
    Prevalence                 0.012579          0.003145      0.003145
    Detection Rate             0.009434          0.000000      0.000000
    Detection Prevalence       0.009434          0.000000      0.000000
    Balanced Accuracy          0.875000          0.500000      0.500000
                         Class: AETERM Class: AETOXGR Class: DROP Class: DTHDTC
    Sensitivity                1.00000             NA      0.9951            NA
    Specificity                1.00000              1      0.5175             1
    Pos Pred Value             1.00000             NA      0.7868            NA
    Neg Pred Value             1.00000             NA      0.9833            NA
    Prevalence                 0.01258              0      0.6415             0
    Detection Rate             0.01258              0      0.6384             0
    Detection Prevalence       0.01258              0      0.8113             0
    Balanced Accuracy          1.00000             NA      0.7563            NA
                         Class: FAORRES Class: QNAM_AESI Class: QNAM_COPD
    Sensitivity                      NA               NA               NA
    Specificity                       1                1                1
    Pos Pred Value                   NA               NA               NA
    Neg Pred Value                   NA               NA               NA
    Prevalence                        0                0                0
    Detection Rate                    0                0                0
    Detection Prevalence              0                0                0
    Balanced Accuracy                NA               NA               NA
                         Class: QNAM_ESAM1 Class: QNAM_ESAM2 Class: QNAM_ESAM3
    Sensitivity                   0.000000          0.000000                NA
    Specificity                   1.000000          1.000000                 1
    Pos Pred Value                     NaN               NaN                NA
    Neg Pred Value                0.996855          0.996855                NA
    Prevalence                    0.003145          0.003145                 0
    Detection Rate                0.000000          0.000000                 0
    Detection Prevalence          0.000000          0.000000                 0
    Balanced Accuracy             0.500000          0.500000                NA
                         Class: QNAM_EXER Class: QNAM_EXLAB Class: QNAM_EXSAB
    Sensitivity                        NA                NA                NA
    Specificity                         1                 1                 1
    Pos Pred Value                     NA                NA                NA
    Neg Pred Value                     NA                NA                NA
    Prevalence                          0                 0                 0
    Detection Rate                      0                 0                 0
    Detection Prevalence                0                 0                 0
    Balanced Accuracy                  NA                NA                NA
                         Class: QNAM_EXSTER Class: SITEID Class: STUDYID
    Sensitivity                          NA       0.00000        0.00000
    Specificity                           1       1.00000        1.00000
    Pos Pred Value                       NA           NaN            NaN
    Neg Pred Value                       NA       0.98742        0.98742
    Prevalence                            0       0.01258        0.01258
    Detection Rate                        0       0.00000        0.00000
    Detection Prevalence                  0       0.00000        0.00000
    Balanced Accuracy                    NA       0.50000        0.50000
                         Class: SUBJID Class: SUPPAE.QVAL
    Sensitivity                1.00000                 NA
    Specificity                1.00000                  1
    Pos Pred Value             1.00000                 NA
    Neg Pred Value             1.00000                 NA
    Prevalence                 0.01258                  0
    Detection Rate             0.01258                  0
    Detection Prevalence       0.01258                  0
    Balanced Accuracy          1.00000                 NA


### Drop Vars


```R
dropvarsAE <- c("projectid","studyid","environmentName","subjectId","StudySiteId",
              "siteid","Site","SiteGroup","instanceId","InstanceName","InstanceRepeatNumber","folderid","Folder","FolderName",
              "FolderSeq","TargetDays","DataPageId","DataPageName","PageRepeatNumber","RecordDate","RecordId","recordposition",
              "RecordActive","SaveTs","MinCreated","MaxUpdated")

dropvarAE<-data.frame()
```
