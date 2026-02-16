
#Dataset_preprocess
#Dependencies: pandas, numpy, librosa, soundfile



#This is for the raw+spetrogram versions, might need separate for opensmile
#Call the dataset_preprocess() like so : dataset_preprocess emodb 16000
#dataset names are emodb iemocap ravdess and savee - lower case
#Run this script from the root folder containing the four dataset folders in their original strucuture
#Ie .\this_script.py
#     /EmoDB/
#     /SAVEE/
#etc. Slashes need to be changed to forward slashes if running on windows. But the training process requires linux anyway */ 
import sys
import os 
import pandas as pd
import numpy as np
import librosa as lr
import librosa.display
import soundfile as sf
#Arguments 
#python dataset_preprocess.py emodb 16000 4 y

#takeargs
which_dataset = sys.argv[1]
SAMPLE_RATE = int(sys.argv[2])
SAMPLE_DURATION = int(sys.argv[3])
z_score = sys.argv[4] #Flag for performing z-score normalisation at the data preprocessing stage. This should be y or n 

#constants for Mel/MFCC creation
  
FRAME_WIDTH = 512 # increase to 512 now 
NUM_SPECTROGRAM_BINS = 512 # 512 is recommended for speech (default is 2048 and suited for music)
NUM_MEL_BINS = 128
LOWER_EDGE_HERTZ = 80.0 # Human speech is not lower
UPPER_EDGE_HERTZ = 7600.0 # Higher is inaudbile to humans   
N_MFCC = 40

#Dataset preprocessing 
# Go through dataset and convert sample rate, Trim or pad to a uniform duration, normalise via zero mean and unit variance 
# Create the Mel spectrogram and MFCCs from the duration-adjusted and normalised samples
# Split dataset into training/validation/test 

#To preprocess I iterate through a CSV, The csv's are provided with this script so they do not need to be rebuilt, however they include 
# directory paths and so this is why the original structure of the dataset needs to be preserved
# The CSV needs to be in same root folder as this script 
#Should specify in readme these folders should be for IEMOCAP "IEMOCAP_full_release", etc, to avoid confusion. 


if which_dataset == "emodb":
     DATASET_PATH = "EmoDB/wav"
     os.path.join("EMODB", "wav")
     dataset_name = "EmoDB" # Simply so the cmdline argument need not be caps-sensitive, but this way it preserves the capitalisation of the original dataset. 
     csv_path = "emodb.csv"        
     
elif which_dataset == "iemocap":
     DATASET_PATH = "IEMOCAP_full_release"
     dataset_name = "IEMOCAP"
     csv_path = "iemocap.csv"
     
elif which_dataset == "ravdess":
     DATASET_PATH = "RAVDESS"
     dataset_name = "RAVDESS"
     csv_path = "ravdess.csv"
     
elif which_dataset == "savee":
     DATASET_PATH = "SAVEE/AudioData"
     dataset_name = "SAVEE"
     csv_path = "savee.csv"
     
     
else:
     print("Incorrect dataset provided, options are: emodb iemocap ravdess savee")


#Setting up
dataset_path = os.path.join(os.getcwd(), DATASET_PATH) 
if z_score == 'y':
     out_path = os.path.join(dataset_name, 'norm_and_fixedduration')
else: 
     out_path = os.path.join(dataset_name, '_fixedduration')

df = pd.read_csv(csv_path)

# --- Top-level utilities (lifted from nested functions to simplify testing and reuse) ---
DATASET_SPEAKER_DEFAULTS = {
     'emodb': {'val_speaker': [3], 'test_speaker': [8]},
     'iemocap': {'val_speaker': ['1_F'], 'test_speaker': ['1_M']},
     'ravdess': {'val_speaker': ['Actor_01', 'Actor_02'], 'test_speaker': ['Actor_03', 'Actor_04']},
     'savee': {'val_speaker': ['DC'], 'test_speaker': ['JE']},
}

def split_data(df, which_dataset, out_dir=None, val_speaker=None, test_speaker=None,
               speaker_column='speaker', filename_columns=None):
     """Split dataframe by speaker into train/val/test and save CSVs.

     This enforces speaker-based splits only (no fraction fallback). Defaults are
     taken from DATASET_SPEAKER_DEFAULTS and must be edited in-code for
     reproducibility of experiments.
     """
     import os

     if filename_columns is None:
          filename_columns = ['file', 'mel_spectrogram', 'MFCCs', 'duration_adjusted']

     if out_dir is None:
          out_dir = os.path.join(which_dataset, 'data')
     os.makedirs(out_dir, exist_ok=True)

     ds_key = which_dataset.lower() if isinstance(which_dataset, str) else which_dataset
     ds_defaults = DATASET_SPEAKER_DEFAULTS.get(ds_key, {})
     if val_speaker is None:
          val_speaker = ds_defaults.get('val_speaker')
     if test_speaker is None:
          test_speaker = ds_defaults.get('test_speaker')

     if val_speaker is None or test_speaker is None:
          raise ValueError("Speaker-based split required. Provide val_speaker and test_speaker or add dataset defaults.")

     if speaker_column not in df.columns:
          raise KeyError(f"Expected speaker column '{speaker_column}' in dataframe. Found columns: {list(df.columns)}")

     def _to_list(x):
          if x is None:
               return []
          if isinstance(x, (list, tuple, set)):
               return list(x)
          return [x]

     val_list = _to_list(val_speaker)
     test_list = _to_list(test_speaker)

     excluded = set(val_list + test_list)
     df_train = df[~df[speaker_column].isin(excluded)].reset_index(drop=True)
     df_val = df[df[speaker_column].isin(val_list)].reset_index(drop=True)
     df_test = df[df[speaker_column].isin(test_list)].reset_index(drop=True)

     train_csv = os.path.join(out_dir, 'train.csv')
     val_csv = os.path.join(out_dir, 'val.csv')
     test_csv = os.path.join(out_dir, 'test.csv')

     df_train.to_csv(train_csv, index=False)
     df_val.to_csv(val_csv, index=False)
     df_test.to_csv(test_csv, index=False)

     print(f"Train samples: {len(df_train)} -> {train_csv}")
     print(f"Validation samples: {len(df_val)} -> {val_csv}")
     print(f"Test samples: {len(df_test)} -> {test_csv}")

     return df_train, df_val, df_test


def listwavs(dataframe):
     """Return a list of numpy arrays loaded from paths listed in dataframe['file'].

     This function expects dataset_path and SAMPLE_RATE to be defined at module scope
     (as they are in this script).
     """
     list_wavs = []
     for file in dataframe['file']:
          audio_file_path = os.path.join(dataset_path, file[4:])
          print("audio file path: ", audio_file_path)
          x, _ = lr.load(audio_file_path, sr=SAMPLE_RATE)
          list_wavs.append(x)
     return list_wavs


def trim_wave(wave):
     duration = int(SAMPLE_DURATION) * SAMPLE_RATE
     return wave[0:duration]


def pad_wave(wave):
     duration = int(SAMPLE_DURATION) *  SAMPLE_RATE
     padding = int(duration - len(wave))
     if padding <= 0:
          return wave
     return np.pad(wave, (0, padding), 'constant')


def save_output(wave, filename):
     # Write out audio as 24bit PCM WAV to out_path
     filename = os.path.join(out_path, filename)
     sf.write(filename, wave, SAMPLE_RATE, subtype='PCM_24')


def data_split():
          val_speaker = DATASET_SPEAKER_DEFAULTS[which_dataset]['val_speaker']
          test_speaker = DATASET_SPEAKER_DEFAULTS[which_dataset]['test_speaker']
          # Split based on speaker column
          df_val = df[df['speaker'].isin(val_speaker)].reset_index(drop=True)
          df_test = df[df['speaker'].isin(test_speaker)].reset_index(drop=True)
          df_train = df[~df['speaker'].isin(val_speaker)].reset_index(drop=True)
          df_train = df_train[~df_train['speaker'].isin(test_speaker)].reset_index(drop=True)


          df_train.to_csv('train.csv', index=False)
          print(df_train.isna().sum())  # total NaNs per column)
          df_val.to_csv('val.csv', index=False)
          print(df_val.isna().sum())  # total NaNs per column)
          df_test.to_csv('test.csv', index=False)
          print(df_test.isna().sum())  # total NaNs per column)

          print(f"Train samples: {len(df_train)}")
          print(f"Validation samples (speaker {val_speaker}): {len(df_val)}")
          print(f"Test samples (speaker {test_speaker}): {len(df_test)}")

          print(df['emotion'].value_counts())
          print(df_train['emotion'].value_counts())
          print(df_val['emotion'].value_counts())
          print(df_test['emotion'].value_counts())

          print(len(df))
          print(len(df_train))
          print(len(df_val))
          print(len(df_test))

data_split()     

# Defining the functions for dataset preprocessing
def norm_script():     

     #Z-score normalisation
     
     
     #Now we compute the mean and std from the training data in order to fit
     dataset_path = './' + DATASET_PATH
     globalaudio = np.concatenate(listwavs(pd.read_csv("train.csv")))
     mean = np.mean(globalaudio)
     std = np.std(globalaudio)
     print("Progress: global values for z-score normalisation calculated")


     #Applying functions
    
     for file in os.listdir(dataset_path):
          print("file =", file)

          audio_file = os.path.join(dataset_path,file)        
          print("audio file = ", audio_file)
          y,sr = lr.load(audio_file,sr=SAMPLE_RATE)  
          #Normalise via zero mean and 1 unit variance if z-score at this stage is selected
          if z_score == 'y':
               y_norm =  (y - mean) / std
          else: 
               y_norm = y #Use the original sample instead of the normalised
          if not os.path.exists(out_path):
               os.makedirs(out_path)
          if lr.get_duration(y=y,sr=sr) > SAMPLE_DURATION:
               trimmed_wave = trim_wave(y_norm)
               save_output(trimmed_wave,file)
               print(file," saved")
          else: 
               padded_wave = pad_wave(y_norm)
               save_output(padded_wave,file)
               print(file," saved")


     for index, file in enumerate(df['file'].values):   
          df.loc[index, 'duration_adjusted'] = os.path.join(out_path, str(file))        
          df.to_csv(which_dataset + "preprocessed.csv")

     print("Progress: Z-score normalisation and fixing of sample duration completed")


     #Creating Mel and MFCCs. 

   

     dataset_path = DATASET_PATH  #Take the trimmed/padded sound files 
     mel_path = os.path.join(out_path, "mel")
     mfcc_path = os.path.join(out_path, "mfccs")
     if not os.path.exists(mel_path):
          os.makedirs(mel_path)
     if not os.path.exists(mfcc_path):
          os.makedirs(mfcc_path)
     

     for file in os.listdir(dataset_path):
          audio_file = os.path.join(dataset_path,file)    
          print(str(audio_file))
          samples, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
    
          #Create spectrogram
          sgram = librosa.stft(samples,n_fft=NUM_SPECTROGRAM_BINS)  


          # use the mel-scale instead of raw frequency on
          sgram_mag, _ = librosa.magphase(sgram)
          mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, n_fft= FRAME_WIDTH,
                                                       sr=sample_rate,fmin=LOWER_EDGE_HERTZ,fmax=UPPER_EDGE_HERTZ,
                                                       n_mels = NUM_MEL_BINS)
          librosa.display.specshow(mel_scale_sgram)

          # use the decibel scale to get the final Mel Spectrogram, as the human hear perceives loudness this way
          mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min) 
          #Now we should have an actual mel spectrogram
          np.save(os.path.join(mel_path, str(file)[:-4]),mel_sgram)
          mfccs = librosa.feature.mfcc(S=mel_sgram,sr=SAMPLE_RATE,n_mfcc=N_MFCC)
          np.save(os.path.join(mfcc_path, str(file)[:-4]),mfccs)      

     #Write path to the mel-spectrograms and mffcs for each utterance to the appropriate row in the CSV
     for index, file in enumerate(df['file'].values):   
          df.loc[index, 'mel_spectrogram'] = os.path.join(mel_path,str(file)[4:-4]+".npy")
          df.loc[index, 'MFCCs'] = os.path.join(mfcc_path, str(file)[4:-4]+".npy")

     df.to_csv(which_dataset + "preprocessed_with_mel_mfcc.csv")


     

norm_script()