# Introduction

Speech Emotion Recognition (SER) is a challenging task in machine learning, and the datasets are too dissimilar for a one-size-fits-all model to allow for a uniform comparison. A single Logistic Regression model was used as a baseline for traditional machine learning approaches with minimal preprocessing, as openSMILE feature extraction produces fixed-length feature vectors regardless of input duration. While the raw audio samples were variable in length, the use of a maximum duration parameter of 4 seconds removed the need for explicit duration normalisation. In contrast, deep learning models require uniform input sizes, and as a result, fixed-length samples are standard practice in the domain.

Across all datasets, model architecture was driven primarily by the input representation rather than the dataset itself. Models trained on raw waveforms consistently used eight convolutional layers, reflecting the high dimensionality of the input. Mel spectrogram and MFCC models used only four convolutional layers due to lower input dimensionality, but a Dropout layer was frequently applied in Mel/MFCC models for regularisation, and dense layers were occasionally added, as in the RAVDESS models. In the case of RAVDESS, which contains a relatively large number of speakers (24), the dense layer provided additional capacity to capture more complex patterns in a speaker-independent setting. Because test speakers were never seen during training, any speaker-correlated patterns learned by the network contribute positively to generalisation, helping the model distinguish emotions across unseen speakers without overfitting to individual identities.

Larger input durations have been shown to improve accuracy up to a certain point, after which further increases yield negligible gains. Preliminary model architectures were created using shorter sample durations to hasten training, before increasing the duration based on the mean sample length of each dataset. The IEMOCAP waveform model degraded noticeably when the sample duration exceeded five seconds, whereas a longer duration of seven seconds was beneficial when using Mel spectrogram and MFCC representations.

With respect to batch size, He, Liu and Tao (2019) showed that batch size had a negative correlation with generalisability when the learning rate remains fixed. A batch size of 16 was chosen in the interest of training speed, except in some cases a model performed uncharacteristically poor, and reducing the batch size to 4 led to significant improvements.

The number of training epochs was set to 50 for waveform models, as their larger input size led to diminishing returns or performance degradation beyond this point. The reduced dimensionality of Mel spectrogram and MFCC inputs allowed for 100 epochs. For the SAVEE dataset, a split of three speakers for training and one for testing prevented the use of a validation set, meaning early stopping and learning rate decay callbacks could not be applied. As a result, training time was reduced to mitigate overfitting, except in the case of the Mel spectrogram model, which did not require this adjustment.

Lower-level architectural details, such as CNN filter and pooling sizes, are not reported here for brevity, though they were carefully tuned for each dataset and input representation.

# DL Models

| Model Name       | Dataset  | Preprocessing                               | Sample rate | Duration | Batch size | Max epochs | Class-Weight Normalisation | Accuracy |
|-----------------|---------|--------------------------------------------|------------|---------|------------|------------|---------------------------|---------|
| emodb_wav       | EmoDB| Z-score normalisation and duration normalisation | 16000      | 4       | 16         | 50         | 1                         | 72.41% |
| emodb_mel       | EmoDB| Z-score normalisation and duration normalisation | 16000      | 4       | 4         | 100        | 0                         | 67.94% |
| emodb_mfcc      | EmoDB   | Z-score normalisation and duration normalisation | 16000      | 4       | 4          | 100        | 0                         | 89.67% |
| emodb_ensemble  | EmoDB   | N/A                                        | N/A        | N/A     | N/A        | N/A        | N/A                       | 87.93% |
| ravdess_wav     | RAVDESS | duration normalisation                      | 16000      | 4       | 16         | 50         | 1                         | 46.67% |
| ravdess_mel     | RAVDESS | Z-score normalisation and duration normalisation | 16000      | 4       | 16         | 100        | 1                         | 60.00% |
| ravdess_mfcc    | RAVDESS | Z-score normalisation and duration normalisation | 16000      | 4       | 16         | 100        | 1                         | 53.33% |
| ravdess_ensemble| RAVDESS | N/A                                        | N/A        | N/A     | N/A        | N/A        | N/A                       | 62.50% |
| savee_wav       | SAVEE   | Z-score normalisation and duration normalisation | 16000      | 5       | 16         | 20         | 0                         | 52.50% |
| savee_mel       | SAVEE   | Z-score normalisation and duration normalisation | 16000      | 5       | 16         | 100        | 1                         | 45.83% |
| savee_mfcc      | SAVEE   | Z-score normalisation and duration normalisation | 16000      | 5       | 4          | 46         | 0                         | 44.17% |
| savee_ensemble  | SAVEE   | N/A                                        | N/A        | N/A     | N/A        | N/A        | N/A                       | 62.50% |
| iemocap_wav     | IEMOCAP | Only fixed duration normalisation           | 16000      | 5       | 16         | 50         | 1                         | 52.53% |
| iemocap_mel     | IEMOCAP | Z-score normalisation and duration normalisation | 16000      | 7       | 16         | 100        | 1                         | 45.68% |
| iemocap_mfcc    | IEMOCAP | Z-score normalisation and duration normalisation | 16000      | 7       | 4          | 100        | 1                         | 46.43% |
| iemocap_ensemble| IEMOCAP | N/A                                        | N/A        | N/A     | N/A        | N/A        | N/A                       | 50.45% |

# References:
He, F., Liu, T. and Tao, D. (2019) Control Batch Size and Learning Rate to Generalize Well: Theoretical and Empirical Evidence. Available at: https://proceedings.neurips.cc/paper/2019/file/dc6a70712a252123c40d2adba6a11d84-Paper.pdf 
