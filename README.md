# Speaker-Independent Speech Emotion Recognition
For the final project of my Bachelor's of Science in Computing & IT, I worked on a Machine Learning (ML) project entitled "Investigating Machine Learning and Deep Learning approaches to Speech Emotion Recognition".

This project was created to verify the integrity of my results, and as an updated and refactored version of the original implementation. The project has been refactored and reorganised, allowing the same components to work across the different dataset, and with command line arguments modifying hyperparameter values. I hope it can be of assistance to others embarking undertaking ML-based or ML-assisted research, and especially Speech Emotion Recognition (SER) research. 

# Instructions

All scripts are designed to be cross-platform. However as of now tensorflow is effectively linux-only due to CUDA compatability, if you circumvent this by disabling such functionality you may see different results to those purported here.

1. Download the SER dataset(s), and download this repository. 
   
2. Extract the dataset(s) into your project's "datasets" directory, preserving the original dataset structure as follows: 
```
Project_root/datasets/EmoDB/wav/audiofile.wav
Project_root/datasets/IEMOCAP/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/Session1/etc...
Project_root/datasets/SAVEE/AudioData/JE/audiofile.wav 
Project_root/datasets/RAVDESS/Actor_01/audiofile.wav the upper-most folder for each each dataset your project directory
```
Rather than search for audio files by name, the scripts will iterate over a CSV file which contains a list of all audio samples used, including a path to said file. This is because the directory structure of certain datasets can be quite complex, as they denote dataset metadata such as the speaker identity, or the emotion that the particular sample conveys.

Therefore, It is essential this step is performed so your dataset structure is as above. This is because file paths are used to directly access audio samples, and the path to said filenames needs to match that of the paths in the CSV file for each dataset. 

For example, make sure you do not have duplicate parent directory for a dataset such as "EmoDB/EmoDB/wav/etc" and IEMOCAP's root directory is IEMOCAP and not "IEMOCAP_full_release_without..."
For SAVEE,RAVDESS and EmoDB you can simply exctract the dataset archives you obtained in step 1 into the datasets folder and the structure will be identical without any user intervention required.

Special note for IEMOCAP: You should create an "IEMOCAP" directory inside your "datasets" folder and extract the dataset there. Depending on the version of IEMOCAP you have acquired, such as the version including video, you may need to modify the upper level folders slightly, this will be extremely simple, yet easier and safer than modifying the file paths stored in each row in the iemocap.csv that dataset_preprocess.py refers to. Simply refer to the structure I have outlined above for IEMOCAP, and make that your own is identical.

3. Download the csv files relevant for the datasets you are interested in, including: emodb.csv iemocap.csv savee.csv and ravdess.csv, place these csv files in your project directory
on the same level as the dataset folders.
Do not modify the original dataset's files or directory structure

4. To create the baseline models using the traditional ML approach with openSMILE, run train_baselines.py
See train_baselines.py for instructions and commandline argument explanations. 

5. Deep Learning model preprocesssing and training To test a model, you must first preprocess the dataset
   This is performed via preprocess_dataset.py, the arguments required are given and explained within that script.
   
6. Train the model using train_model.py, again from the commandline

7. Finally, once you have have trained mulitple models, you can create and test an post-decision ensemble using ensemble.py
   The training script will have saved predictions in a fixed format and  these will be located and used by ensemble.py depending
   on the models specified.


train_model.py and ensemble.py will print results to the command line, and a confusion matrix will also be saved to confusion_matrices folder created in your working directory for you to view.


### Dependencies 
See requirements.txt


# Project Summary 
The project compared the traditional ML approach to SER with that of the modern Deep Learning (DL) approach. The study stands out amongst many contemporary work in the domain by virtue of being speaker-independent. That is, the model's training, validation and test data were segmented by speaker rather than purely by a arbritary percentage of the total dataset size. As a result, the model was not exposed to audio samples from the same voice that featured in validation or test data. This is because peaker-specific qualities "leak" into the learning process in speaker-dependent studies where training data includes samples from all speakers in the total dataset. Speaker-independent models usually do not perform on par with speaker-independent models, but an SER model can not be considered truly generalisable if it is speaker-dependent. 

The traditional ML approach was that of utilising handcrafted features, in this case I used OpenSMILE (open-source Speech and Music Interpretation by Large-space Extraction) to extract features according to the emobase feature set, and these features were then used to train a logistic regression classifier. 

For the DL approach, I utilised a simple approach relative to the state of the art, my project utilises predominantly Convolutional Neural Networks (CNN), however I trained on three data representations separately, the raw waveforms (WAV), Mel Spectrograms, and Mel Frequency Cepstrum Coefficients (MFFCs). During my work I noticed that different models, especially from each of the different data representations, would often excel or struggle especially with certain emotions in particular, and it was rare that all the emotions were learned with a similar level of success. I therefore create a "post-fusion" or "late-fusion" ensemble model, by aggregating the class-probabilties (predictions) of the best-performing models for each of the data representations. In this way I demonstrated that a simple aggregate model can improve upon my single best-performing individual models. 

Results were compared across the four datasets, between the CNN models and the different data representations, and between the Dland traditional ML classifier.

Most of the time on my project was spent researching, and experimenting with data preprocessing optimisations, for example regarding neural network architecture optimisations for the CNNs, and after multiple iterations the DL approach outperformed the traditional ML approach despite it's leveraging of heuristic features. 

Four datasets for SER were used, EmoDB, IEMOCAP, SAVEE, and RAVDESS, and between them I try and capture a balance and variety in dataset attributes. 

In conclusion, results support the view that deep learning techniques achieve superior performance on speaker-independent speech emotion recognition compared to traditional ML approaches that rely on domain knowledge via handcrafted audio features. However, this study did not use handcrafted features at all in a DL model, and yet this approach has been shown to be succesful, in both standalone models and as part of an ensemble with other DL models using different data representations.

# Datasets used in this Project
## Dataset Characteristics

| Attribute | Emo-DB | SAVEE | IEMOCAP | RAVDESS |
|---|---|---|---|---|
| Full Name | Berlin Database of Emotional Speech | Surrey Audio-Visual Expressed Emotion Database | Interactive Emotional Dyadic Motion Capture Database | Ryerson Audio-Visual Database of Emotional Speech and Song |
| Year Released | 2005 | 2007 | 2008 | 2018 |
| Language | German | English | English | English |
| Modality | Audio only | Audio + Video | Audio + Video + Motion Capture | Audio + Video |
| Labels | Emotion: anger, boredom, disgust, fear, happiness, sadness, neutral | Emotion: anger, boredom, disgust, fear, happiness, sadness, neutral | Emotion: neutral, frustrated, happy, sad, angry, excited, surprised, other, fearful, disgusted<br>Includes confidence score (0–1) | Emotion: calm, happy, sad, angry, fearful, surprise, disgust<br>Emotion intensity: normal, strong |
| Speakers | 10 actors (5 male, 5 female) | 4 male actors | 10 actors (5 male, 5 female) | 24 actors (12 male, 12 female) |
| Emotion Types | 7 | 7 | 10 (acted + spontaneous) | 8 |
| Speech Type | Scripted sentences | Scripted sentences | Scripted + spontaneous conversations | Scripted sentences |
| Sampling Rate | 16 kHz | 44.1 kHz | 16 kHz | 48 kHz |
| Total Duration (approx.) | 45 minutes | 1.5 hours | 12 hours | 2.4 hours |
| Introductory Paper | Burkhardt, 2005 | Jackson, 2008 | Busso, 2008 | Livingstone, 2018 |

# Traditional-ML Model Accuracy 
All classifiers used features extracted from 4 seconds of audio only, at a sample rate of 16000kHz
DL models later used longer samples at times, but at the baseline stage my concern was with parity among the datasets
| Dataset | Accuracy |
| --- | ----
| emoDB | 68.96% |
| RAVDESS | 49.17% |
| SAVEE | 54.16% |
| IEMOCAP | 37.8% |


# Ensemble-Model Accuracy  

| Dataset | Accuracy |
| --- | --- | 
|EmoDB | 88% | 
|RAVDESS | 62.5% |  
|SAVEE | 65% |  
|IEMOCAP | 50.45% |  

For details on the individual models which the ensembles were comprised of, and their hyperparameters, see DL_models.txt in /models/


# Comparison With Contemporary Research

I compared my results with other Speaker Independent (SI) studies. The EmoDB ensemble performs among the top performing speaker independent models, despite using a relatively simple and lightweight approach. I found only one work which outperformed my EmoDB ensemble, Amjad et al. 2021 achieved 92.65% WAR, compared to my model's 91% WAR. Xu et al. 2022 also reported 90.61% Accuracy and Farooq et al. 2020 reported 90.5%. 

The other ensembles did not perform as well relative to modern research, for RAVDESS the top contemporaries include Amjad et al. 2021 (82.75% WAR), Sayed et al. 2025 (73.75% Acc), and Farooq et al. 2020 (73.5% WAR). All show demonstrably higher accuracy, at around 10-20% higher than my own. 

However, SI studies on RAVDESS were extremely rare, and two of these models used AlexNet pre-trained network, and Sayed used a CNN+LSTM hybrid model which is somewhat more expensive to train. 

My SAVEE ensemble was outperformed by Amjad et al. 2021 (75.38% WAR) and Farooq et al. 2020 (66.90% WAR). This dataset was even rarer as a choice of SI study than RAVDESS. 

IEMOCAP appeared to the most popular dataset for SI SER, and I found the largest number of studies which outperformed my own. My approach seemed to perform relatively poorly on this dataset compared to the others. IEMOCAP is much different to the other three datasets examined, and proves to be among the most challenging of all the SER datasets. The models which outperformed my own all used increasingly more complex techniques such as leveraging pre-trained AlexNets, hybrid feature extraction and feature selection, multi-task learning, pre-trained wav2vec, and multi-head attention. The models I compared with my own that achieved greater accuracies are as follows:

## EmoDB
| Study | Data Representation | Methodology | Evaluation Result | Notes |
|------|--------------------|-------------|------------------|------|
| Amjad et al. 2021 | Mel Spectrogram | Pre-trained AlexNet for feature extraction, CFS for feature selection, traditional ML techniques compared with MLP for classification | 92.65% WAR | MLP classifier performed best for EmoDB |
| Xu et al. 2022 | Mel Spectrogram, MFCC, handcrafted features | Feature vectors from three models concatenated; fourth model trained with fully connected layer | 90.61% Accuracy | |
| Farooq et al. 2020 | Mel Spectrogram | AlexNet feature extraction, CFS feature selection, classification via MLP or traditional ML | 90.50% WAR | MLP gave best result |
| This project | Raw waveform, Mel Spectrogram, MFCC | Ensemble of three CNN models separately trained on each representation | 87.93% Accuracy | |
| Sayed et al. 2025 | Wavelet Scaled Spectrogram | CNN + LSTM | 87.78% Accuracy | Mean of 3 folds |
| Human | — | — | 86% Average Unweighted Accuracy | Mean from human evaluation results (Burkhardt, 2005) |
| Meng et al. 2019 | 3D Mel Spectrogram | 3D Dilated CNN + BiLSTM + Residual Block | 85.39% Accuracy | |
| Zhao, Mao and Chen 2019 | Mel Spectrogram | CNN + LSTM | 82.42% Accuracy | Raw waveform only achieved 57% |
| Xu et al. 2024 | Spectrogram | CNN for local features, GRU for global features, multi-head attention integration | 80.2% UAR | |
| Rintala 2024 | Raw waveform | CNN + LSTM with parallel branches | 75.78% Accuracy | Based on Latif (2019) architecture |

## RAVDESS
| Study | Data Representation | Methodology | Evaluation Result | Notes |
|------|--------------------|-------------|------------------|------|
| Amjad et al. 2021 | Mel Spectrogram | Pre-trained AlexNet feature extraction, CFS feature selection, classification with MLP or traditional ML | 82.75% WAR | MLP performed best |
| Sayed et al. 2025 | Wavelet Scaled Spectrogram | CNN + LSTM | 73.75% Accuracy | Average of 3 folds |
| Farooq et al. 2020 | Mel Spectrogram | AlexNet feature extraction + CFS + ML classifiers | 73.50% | MLP gave best result |
| This project| Raw waveform, Mel Spectrogram, MFCC | Ensemble of three CNN models trained on each representation | 62.5% | |
| Rintala 2024 | Raw audio | CNN + LSTM with parallel branches | 61.67% Accuracy | Based on Latif (2019) |
| Human | — | — | 62% Accuracy | |

## SAVEE
| Study | Data Representation | Methodology | Evaluation Result | Notes |
|------|--------------------|-------------|------------------|------|
| Amjad et al. 2021 | Mel Spectrogram | Pre-trained AlexNet feature extraction, CFS feature selection, classification via MLP | 75.38% WAR | Best result obtained with MLP |
| Farooq et al. 2020 | Mel Spectrogram | AlexNet feature extraction + CFS + ML classifiers | 66.90% WAR | MLP performed best |
| Human | — | — | 66.5% Average Accuracy | |
| This project | Raw waveform, Mel Spectrogram, MFCC | Ensemble of three CNN models trained on different representations | 65% | Achieved comparable performance with human, but it seems this dataset is too small and with too few speakers for a DL model relying on large datasets|
| Sivanagaraja et al. 2017 | Raw waveform | Multiple convolution branches with different downsampling scales | 50.28% Accuracy | |

## IEMOCAP
| Study | Data Representation | Methodology | Evaluation Result | Notes |
|------|--------------------|-------------|------------------|------|
| Amjad et al. 2021 | Mel Spectrogram | Pre-trained AlexNet feature extraction, CFS feature selection, classification via ML algorithms | 84% WAR | SVM performed best |
| Cai et al. 2021 | wav2vec 2.0 | Multi-task learning using pre-trained wav2vec 2.0 encoding | 78.15% WAR | |
| Farooq et al. 2020 | Mel Spectrogram | AlexNet feature extraction + CFS + ML classifiers | 73.50% WAR | SVM gave best result |
| Xu et al. 2022 | Mel Spectrogram, MFCC, handcrafted features | Feature concatenation from three models with additional fully connected model | 73.42% Unweighted Accuracy | |
| Xu et al. 2024 | Spectrogram | GRU with multi-head attention | 70.2% UAR | |
| Chen 2018 | 3D Mel Spectrogram | CNN + LSTM + Attention | 64.74% UAR | |
| Fayek et al. 2017 | Raw waveform | CNN end-to-end | 60.89% UAR | Frame-based approach performed best |
| Latif 2019 | Raw waveform | CNN + LSTM + DNN | 60.23% UAR | Early parallel-branch SER architecture |
| Vladimir Chernykh 2018 | MFCC + Chroma features | Frame-wise Bi-LSTM | 54% Accuracy | |
| This project | Raw waveform, Mel Spectrogram, MFCC | Ensemble of three CNN models trained on each representation | 50.45% | The most popular dataset for DL-SER research, the most complex and natural dataset seems to require more complex and intensive approaches than my own|
| Han, K., Yu, D. and Tashev, I. 2014 | Handcrafted features (MFCC, pitch, delta) | DNN feature extraction + ELM classifier | 48.2% UAR | |




###References
Amjad, A., Khan, L. and Chang, H.-T. (2021) ‘Effect on speech emotion classification of a feature selection approach using a convolutional neural network’, PeerJ Computer Science, 7, p. e766. https://doi.org/10.7717/peerj-cs.766
Cai, X. et al. (2021) ‘Speech Emotion Recognition with Multi-Task Learning’, Interspeech 2021. https://doi.org/10.21437/interspeech.2021-1852
Chen, M. et al. (2018) ‘3-D Convolutional Recurrent Neural Networks With Attention Model for Speech Emotion Recognition’, IEEE Signal Processing Letters, 25(10), pp. 1440–1444. https://doi.org/10.1109/lsp.2018.2860246
Chernykh, V. and Prikhodko, P. (2018) ‘Emotion Recognition from Speech with Recurrent Neural Networks’, arXiv preprint. https://doi.org/10.48550/arXiv.1701.08071
Farooq, M. et al. (2020) ‘Impact of Feature Selection Algorithm on Speech Emotion Recognition Using Deep Convolutional Neural Network’, Sensors, 20(21), p. 6008. https://doi.org/10.3390/s20216008
Fayek, H.M., Lech, M. and Cavedon, L. (2017) ‘Evaluating deep learning architectures for Speech Emotion Recognition’, Neural Networks, 92, pp. 60–68. https://doi.org/10.1016/j.neunet.2017.02.013
Han, K., Yu, D. and Tashev, I. (2014) ‘Speech Emotion Recognition Using Deep Neural Network and Extreme Learning Machine’, Interspeech 2014. https://doi.org/10.21437/interspeech.2014-57
Latif, S. et al. (2019) ‘Direct Modelling of Speech Emotion from Raw Speech’, arXiv preprint. https://doi.org/10.48550/arxiv.1904.03833
Meng, H. et al. (2019) ‘Speech Emotion Recognition From 3D Log-Mel Spectrograms With Deep Learning Network’, IEEE Access, 7, pp. 125868–125881. https://doi.org/10.1109/access.2019.2938007
Rintala, J. (2024) ‘CNN + LSTM with Parallel Branches for Speech Emotion Recognition’, DiVA Portal. https://www.diva-portal.org/smash/get/diva2:1456228/FULLTEXT01.pdf
Sayed, N. et al. (2025) ‘A Combine CNN-RNN Based Approach for Augmenting the Performance of Speech Emotions Recognition’, Traitement du Signal, 42(2), pp. 647–662. https://doi.org/10.18280/ts.420205
Sivanagaraja, T. et al. (2017) ‘End-to-End Speech Emotion Recognition Using Multi-Scale Convolution Networks’, DR-NTU, Nanyang Technological University. https://doi.org/10.1109/apsipa.2017.8282026
Xu, C. et al. (2024) ‘A New Network Structure for Speech Emotion Recognition Research’, Sensors, 24(5), p. 1429. https://doi.org/10.3390/s24051429
Xu, X. et al. (2022) ‘Multi-Type Features Separating Fusion Learning for Speech Emotion Recognition’, Applied Soft Computing, 130, p. 109648. https://doi.org/10.1016/j.asoc.2022.109648
Zhao, J., Mao, X. and Chen, L. (2019) ‘Speech Emotion Recognition Using Deep 1D & 2D CNN LSTM Networks’, Biomedical Signal Processing and Control. https://doi.org/10.1016/j.bspc.2018.08.035


