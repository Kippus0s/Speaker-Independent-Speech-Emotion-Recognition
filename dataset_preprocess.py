
#Dataset_preprocess
#Dependencies: pandas, numpy, librosa, soundfile



#This is for the raw+spetrogram versions, might need separate for opensmile
#Call the dataset_preprocess() like so : dataset_preprocess emodb 16000
#dataset names are emodb iemocap ravdess and savee - lower case
#Run this script from the root folder containing the four dataset folders in their original strucuture
#Ie .\this_script.py
#     /EmoDB/
#     /SAVEE/

from importlib.resources import files
import sys
import os 
import pandas as pd
import numpy as np
import librosa as lr
import librosa.display
import soundfile as sf

#Arguments 
# ex. " python dataset_preprocess.py emodb 16000 4 y " 

# takeargs from command line, which dataset, sample rate, sample duration, whether to z-score normalise or not (y/n)
# With exception handling and attempts to alert user of incorrect arguments

try:
    which_dataset = sys.argv[1]
    SAMPLE_RATE = int(sys.argv[2])
    SAMPLE_DURATION = int(sys.argv[3])
except IndexError:
    print("Missing required arguments")
    sys.exit(1)
except ValueError:
    print("Sample rate and duration must be integers")
    sys.exit(1)


# Checking correct dataset argument name
if which_dataset not in ["emodb", "iemocap", "ravdess", "savee"]:
    raise ValueError(
        "Incorrect dataset provided, options are: emodb, iemocap, ravdess, savee"
    )

print("Dataset selected:", which_dataset)

# Alerting user of sampel rate argument exceeding sample rate of source

if which_dataset == "ravdess":
     if SAMPLE_RATE > 48000:
          flag = input("RAVDESS sample rate of source files are 48khz, this script does not upsample Are you sure you wish to proceed? (y/n)")
     if flag != 'y':
          sys.exit()
     
else:
     if SAMPLE_RATE > 16000:
          flag = input("sample rate of source files are 16khz, this script does not upsample Are you sure you wish to proceed? (y/n)")
          if flag != 'y':
               sys.exit()

# Z_score normalisation argument check
try:
    z_score = sys.argv[4]
except IndexError:
    z_score = 'n'  # Default value if not provided

if z_score not in ['y', 'n']:
     raise TypeError("z-score normalisation must be either 'y' or 'n'")


# Define constants for Mel Spectrogram and MFCC creation
  
FRAME_WIDTH = 512 # increase to 512 now 
NUM_SPECTROGRAM_BINS = 512 # 512 is recommended for speech (default is 2048 and suited for music)
NUM_MEL_BINS = 128
LOWER_EDGE_HERTZ = 80.0 # Human speech is not lower
UPPER_EDGE_HERTZ = 7600.0 # Higher is inaudbile to humans   
N_MFCC = 40


# Dataset preprocessing 
# Go through dataset and convert sample rate, Trim or pad to a uniform duration, normalise via zero mean and unit variance 
# Create the Mel spectrogram and MFCCs from the duration-adjusted and normalised samples
# Split dataset into training/validation/test 

#
# To preprocess I iterate through a CSV, The csv's are provided with this script so they do not need to be rebuilt, however they include 
# directory paths and so this is why the original structure of the dataset needs to be preserved
# The CSV needs to be in same root folder as this script 
#Should specify in readme these folders should be for IEMOCAP "IEMOCAP_full_release", etc, to avoid confusion. 


if which_dataset == "emodb":
     dataset_name = "EmoDB" # Simply so the cmdline argument need not be caps-sensitive, but this way it preserves the capitalisation of the original dataset. 
     data_path = os.path.join(os.getcwd(), dataset_name)
     DATASET_PATH = "EmoDB"
     csv_path = os.path.join(os.getcwd(), "emodb.csv")     
     
elif which_dataset == "iemocap":
     dataset_name = "IEMOCAP"
     data_path = os.path.join(os.getcwd(), dataset_name)
     DATASET_PATH = "IEMOCAP_full_release"     
     csv_path = "iemocap.csv"
     
elif which_dataset == "ravdess":
     dataset_name = "RAVDESS"
     data_path = os.path.join(os.getcwd(), dataset_name)
     DATASET_PATH = data_path
     csv_path = "ravdess.csv"
     
elif which_dataset == "savee":
     dataset_name = "SAVEE"
     data_path = os.path.join(os.getcwd(), dataset_name)
     DATASET_PATH = os.path.join(dataset_name, "AudioData")
     csv_path = os.path.join(os.getcwd(), "savee.csv")
     
     
else:
     print("Incorrect dataset provided, options are: emodb iemocap ravdess savee")


#Setting up
dataset_path = os.path.join(os.getcwd(), DATASET_PATH) 
if z_score == 'y':
     out_path = os.path.join(dataset_name, 'norm_and_fixedduration')
else: 
     out_path = os.path.join(dataset_name, '_fixedduration')

df = pd.read_csv(csv_path)

# Validation split constant
# These are the speakers I used for my study, but experiementation with different splits, or cross-validation is welcome
# Rather than perform cross-validation, I performed a simple single-pass speaker-independent validation
# Cross-validation would be preferable, but it is far more time-consuming to complete
# The models went through many iterations and 

DATASET_SPEAKER_DEFAULTS = {
     'emodb': {'val_speaker': [3], 'test_speaker': [8]},
     'iemocap': {'val_speaker': ['1_F'], 'test_speaker': ['1_M']},
     'ravdess': {'val_speaker': ['Actor_01', 'Actor_02'], 'test_speaker': ['Actor_03', 'Actor_04']},
     'savee': {'val_speaker': ['DC'], 'test_speaker': ['JE']},# I concatenate val and train in model config for a 75% train/test split instead.
}


def listwavs(dataframe):
     """Return a list of numpy arrays loaded from paths listed in dataframe['file'].

     This function expects dataset_path and SAMPLE_RATE to be defined at module scope
     (as they are in this script).
     """
     list_wavs = []
     for file in dataframe['file']:
          audio_file_path = audio_file_parser(file)
          x, _ = lr.load(audio_file_path, sr=SAMPLE_RATE)
          list_wavs.append(x)
     return list_wavs

def audio_file_parser(file):
          if dataset_name == "EmoDB":
               audio_file_path = os.path.join(dataset_path, os.path.normpath(file))
          elif dataset_name == "SAVEE":
               audio_file_path = os.path.join(dataset_path, os.path.normpath(file) + ".wav")
               print("normpath(file) gives ", os.path.normpath(file))
          elif dataset_name == "RAVDESS":
               audio_file_path = os.path.join(dataset_path, file[0:8], file[9:])               
          else:
               "finish this soon"
          print("audio file path: ", audio_file_path)
          return audio_file_path

def trim_wave(wave):
     duration = int(SAMPLE_DURATION) * SAMPLE_RATE
     return wave[0:duration]


def pad_wave(wave):
     duration = int(SAMPLE_DURATION) *  SAMPLE_RATE
     padding = int(duration - len(wave))
     if padding <= 0:
          return wave
     return np.pad(wave, (0, padding), 'constant')


def save_output2(wave, filename):
     # Write out audio as 24bit PCM WAV to out_path
     filename = os.path.join(out_path, filename)
     
     sf.write(filename, wave, SAMPLE_RATE, subtype='PCM_24')


def save_output(wave,   filename):

     subdirs = os.path.dirname(os.path.normpath(filename))       
     
     filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"
     
     save_dir = os.path.join(out_path, subdirs)
          
     print("Creating directories:")
     print("save_dir =", save_dir)
     
     try:
          os.makedirs(save_dir, exist_ok=True)
          
     except Exception as e:
          print("Error creating directories:", e)
         
     
     full_path = os.path.join(save_dir, filename)
     sf.write(full_path, wave, SAMPLE_RATE, subtype='PCM_24')

def data_split():
          val_speaker = DATASET_SPEAKER_DEFAULTS[which_dataset]['val_speaker']
          test_speaker = DATASET_SPEAKER_DEFAULTS[which_dataset]['test_speaker']
          # Split based on speaker column
          df_val = df[df['speaker'].isin(val_speaker)].reset_index(drop=True)
          df_test = df[df['speaker'].isin(test_speaker)].reset_index(drop=True)
          df_train = df[~df['speaker'].isin(val_speaker)].reset_index(drop=True)
          df_train = df_train[~df_train['speaker'].isin(test_speaker)].reset_index(drop=True)

          if not os.path.exists(data_path):
               os.makedirs(data_path)
               

          df_train.to_csv(os.path.join(data_path,'train.csv'), index=False)
          print(df_train.isna().sum())  # total NaNs per column)
          df_val.to_csv(os.path.join(data_path,'val.csv'), index=False)
          print(df_val.isna().sum())  # total NaNs per column)
          df_test.to_csv(os.path.join(data_path,'test.csv'), index=False)
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

  

# Defining the functions for dataset preprocessing
def norm_script():     

     #Z-score normalisation
          
     #Now we compute the mean and std from the training data in order to fit
     dataset_path = os.path.join(os.getcwd(), DATASET_PATH)
     globalaudio = np.concatenate(listwavs(pd.read_csv(os.path.join(dataset_name, "train.csv"))))
     mean = np.mean(globalaudio)
     std = np.std(globalaudio)
     print("Progress: global values for z-score normalisation calculated")


     #Applying functions
    
     for file in pd.read_csv(csv_path)['file'].values:
          print("file =", file)

          audio_file = audio_file_parser(file)  
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
               
          else: 
               padded_wave = pad_wave(y_norm)
               save_output(padded_wave,file)
               


     for index, file in enumerate(df['file'].values):   
          df.loc[index, 'duration_adjusted'] = os.path.join(out_path, str(file))        
          df.to_csv(which_dataset + "preprocessed.csv")

     print("Progress: Z-score normalisation and fixing of sample duration completed")


     
def mel_mfcc():
     #Creating Mel and MFCCs. 
     dataset_path = DATASET_PATH  

     # Creating the mel and mfcc directories in the output folder if they do not already exist.
     mel_path = os.path.join(out_path, "mel")
     mfcc_path = os.path.join(out_path, "mfccs")
     if not os.path.exists(mel_path):
          os.makedirs(mel_path)
     if not os.path.exists(mfcc_path):
          os.makedirs(mfcc_path)
     
     # Take the trimmed/padded sound files 
     for file in pd.read_csv(csv_path)['file'].values:

          audio_file = audio_file_parser(file)            
          samples, sample_rate = librosa.load(audio_file, sr=SAMPLE_RATE)
    
          #Create spectrogram
          sgram = librosa.stft(samples,n_fft=NUM_SPECTROGRAM_BINS) 
          print("sgram created")

          # Use the mel-scale instead of raw frequency bins, as the mel scale is more aligned with human perception of sound and is commonly used in speech processing tasks.
          sgram_mag, _ = librosa.magphase(sgram)
          mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, n_fft= FRAME_WIDTH,
                                                       sr=sample_rate,fmin=LOWER_EDGE_HERTZ,fmax=UPPER_EDGE_HERTZ,
                                                       n_mels = NUM_MEL_BINS)
          librosa.display.specshow(mel_scale_sgram)

          # Use the decibel scale to get the final Mel Spectrogram, as the human ear perceives loudness this way
          mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min) 
          
          # Creating the correct file path string to preserve original dataset directory structure in the new folders for the mel spectrograms and MFCCs.
          subdirs = os.path.dirname(os.path.normpath(file))       
          filename = os.path.splitext(os.path.basename(file))[0]  
                  
          mel_save_dir = os.path.join(mel_path, subdirs)
          mfcc_save_dir = os.path.join(mfcc_path, subdirs)
                    
          try:
               os.makedirs(mel_save_dir, exist_ok=True)
               os.makedirs(mfcc_save_dir, exist_ok=True)
          except Exception as e:
               print("Error creating directories:", e)
               continue  # skip this file

          mel_save_path = os.path.join(mel_save_dir, filename + ".npy")  # ensure .npy extension
          np.save(mel_save_path, mel_sgram)
          print("saved mel spectrogram to", mel_save_path)
          mfcc_save_path = os.path.join(mfcc_save_dir, filename + ".npy")
          mfccs = librosa.feature.mfcc(S=mel_sgram,sr=SAMPLE_RATE,n_mfcc=N_MFCC)
          np.save(mfcc_save_path, mfccs)
          print("saved", mfccs, "MFCCs to ", mfcc_save_path)
          
          
     #Write path to the mel-spectrograms and mffcs for each utterance to the appropriate row in the CSV
     for index, file in enumerate(df['file'].values):   
          df.loc[index, 'mel_spectrogram'] = os.path.join(mel_path,str(file)[4:-4]+".npy")
          df.loc[index, 'MFCCs'] = os.path.join(mfcc_path, str(file)[4:-4]+".npy")

     df.to_csv(which_dataset + "preprocessed_with_mel_mfcc.csv")

     
data_split()   
norm_script()
mel_mfcc()