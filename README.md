# Speaker-Independent Speech Emotion Recognition
For the final project of my Bachelor's of Science in Computing & IT, I worked on a Machine Learning (ML) project entitled "Investigating Machine Learning and Deep Learning approaches to Speech Emotion Recognition".

This project is an updated and refactored version, as the original spanned dozens of scripts which were unique to each of the four Speech Emotion Recongition (SER) datasets I utilised.  In addition, this project will verify my results, confirming reproducibility. 

# Project Summary 
The project compared a tradiional ML approach on SER with that of the modern Deep Learning (DL) approach. The study was speaker-independent, meaning that the model's training data was segmented by speaker, in this way the model was not exposed to anaudio samples from the same speaker across splits. This is due to speaker-specific qualities "leaking" into the learning process in speaker-dependent studies, where training data includes samples from all speakers in the total dataset. An SER model can not be considered to be truly generalisable if it is speaker-dependent. 

The traditional ML approach is that of utilising handcrafted features, in this case I used the emobase feature set extracted  with OpenSMILE, which were used to train a logistic regression classifier. 
The DL approach followed, which utilised Convolutional Neural Networks (CNN) trained on three data representations separately, the raw waveforms (WAV), Mel Spectrograms, and Mel Frequency Cepstrum Coefficients (MFFCs).
Ultimately I created a simple aggregation-based ensemble (post-fusion model) for each dataset which aggregated the predicted class probabilities from the models trained on each of the three different data representations. 

Results were compared across the four datasets, between the CNN models and the different data representations, and between the CNN models and traditional ML classifier.
Most of the time was spent experimenting with data preprocessing optimisations, and neural network architecture optimisations for the CNNs, and after multiple iterations the DL approach outperformed the traditional ML approach despite it's leveraging of heuristic features. 

Four of the main datasets for SER were used, EmoDB, IEMOCAP, SAVEE, and RAVDESS, and the work adopted a speaker-independent approach whereby the entirety of at least one speaker's speech samples were excluded from training and used exclusively for testing, in this way there is no bleeding of speaker-specific qualities into the training data. This approach consistently proves to be more challenging for ML models, and it is thought that this improves model generalisability.

My results support the view that deep learning techniques achieve superior performance on speaker-independent speech emotion recognition compared to traditional ML approaches that rely on domain knowledge via handcrafted audio features.

I concluded my project by creating an ensemble model which averaged the softmax probabilities per class from the three separate CNN models trained on different data representations. With exception to one dataset (IEMOCAP) this improved classification accuracy. 

# Traditional ML model accuracy (logistic Regression model trained on openSMILE's emobase feature set)

```
emoDB: 77.6%
RAVDESS 55.8%
SAVEE 51.6%
IEMOCAP: 28%
```

# Ensemble model accuracy  

```
EmoDB: 91%
RAVDESS: 64%
SAVEE: 65%
IEMOCAP: 44% (This ensemble failed to improve the performance, in fact the best performing individual model trained on IEMOCAP was the MFCC model which achieved 48.67%)
```


# Comparison with other contemporary research

I compared my results with other Speaker Independent (SI) studies. The EmoDB ensemble performs among the top performing speaker independent models, despite using a relatively simple and lightweight approach. I found only one work which outperformed my EmoDB ensemble, Amjad et al. 2021 achieved 92.65% WAR, compared to my model's 91% WAR. Xu et al. 2022 also reported 90.61% Accuracy and Farooq et al. 2020 reported 90.5%. 

The other ensembles did not perform as well relative to modern research, for RAVDESS the top contemporaries include Amjad et al. 2021 (82.75% WAR), Sayed et al. 2025 (73.75% Acc), and Farooq et al. 2020 (73.5% WAR). All show demonstrably higher accuracy, at around 10-20% higher than my own. 
However, SI studies on RAVDESS were extremely rare, and two of these models used AlexNet pre-trained network, and Sayed used a CNN+LSTM hybrid model which is somewhat more expensive to train. 

My SAVEE ensemble was outperformed by Amjad et al. 2021 (75.38% WAR) and Farooq et al. 2020 (66.90% WAR). This dataset was even rarer as a choice of SI study than RAVDESS. 

IEMOCAP appeared to the most popular dataset for SI SER, and I found the largest number of studies which outperformed my own. My approach seemed to perform relatively poorly on this dataset compared to the others. IEMOCAP is much different to the other three datasets examined, and proves to be among the most challenging of all the SER datasets. The models which outperformed my own all used increasingly more complex techniques such as leveraging pre-trained AlexNets, hybrid feature extraction and feature selection, multi-task learning, pre-trained wav2vec, and multi-head attention. The models I compared with my own that achieved greater accuracies are as follows:

```
Amjad et al. 2021 (84% WAR)
Cai et al. 2021 (78.15% WAR)
Farooq et al. 2020 (73.50% WAR)
Xu et al. 2022 (73.42 % Unweighted Accuracy)
Xu et al. 2024 (70.2% UAR)
Chen, 2018 (64.74% UAR)
Fayek et al. 2017 (60.89% UAR)
Latif, 2019 (60.23% UAR)
Vladimir Chernykh, 2018 (54% Accuracy)
```


### Dependencies 
See requirements.txt

# Instructions

1. Download the dataset(s).
   
2. Extract the dataset(s) into the same project directory, the upper-most folder for each each dataset your project directory "EmoDB" , "IEMOCAP_full_release_withoutVideos" ,
"RAVDESS" and "SAVEE", which should be the case if you simply extracted the dataset archives directly into your project directory.

3. Download the csv files relevant for the datasets you are interested in, including: emodb.csv iemocap.csv savee.csv and ravdess.csv, place these csv files in your project directory
on the same level as the dataset folders.

Do not modify the original dataset's files or directory structure

4a. >OPENSMILE MODEL PREPROCESSING AND TRAINING>
4b. Deep Learning model preprocesssing and training To test a model, you must first preprocess the dataset by running preprocess_dataset.py in the command line with the following four arguments explained below:


  Command-line arguments:
   ```
   which_dataset      : which dataset to process (incl. emodb, savee, iemocap, ravdess)
   sample_rate       : sampling rate in Hz
   sample_duration   : duration of each sample in seconds
   z_score           : whether to z-score normalize at this preprocessing stage ('y' or 'n')
   --suffix (optional): suffix for output folder name to avoid overwriting previous runs
   
   ex.  python dataset_preprocess.py emodb 16000 4 y 
   or ex. python dataset_preprocess.py savee 16000 3 n --suffix new_output_directory_name
   ```

Next you run the train_model.py with the arguments explained below: 


