# dataset_preprocess.py
from importlib.resources import files
import sys
import os 
import pandas as pd
import numpy as np
import librosa as lr
import librosa.display
import soundfile as sf
import argparse 

# Command-line arguments:
"""
which_dataset      : which dataset to process (incl. emodb, savee, iemocap, ravdess)
sample_rate       : sampling rate in Hz
sample_duration   : duration of each sample in seconds
z_score           : whether to z-score normalize at this preprocessing stage ('y' or 'n')
--output (optional): name for output directory name to avoid overwriting previous runs
ex.  python dataset_preprocess.py emodb 16000 4 y 
or ex. python dataset_preprocess.py savee 16000 3 n --output new_output_directory_name """

# Validation split constants
""" These are the speakers I used for my study, but experiementation with different splits, or cross-validation is welcome
Rather than perform cross-validation, I performed a simple single-pass speaker-independent validation
Cross-validation would be preferable, but it is far more time-consuming to complete, as the models went through many iterations"""

DATASET_SPEAKER_DEFAULTS = {
     'emodb': {'val_speaker': [3], 'test_speaker': [8]},
     'iemocap': {'val_speaker': ['1_F'], 'test_speaker': ['1_M']},
     'ravdess': {'val_speaker': ['Actor_01', 'Actor_02'], 'test_speaker': ['Actor_03', 'Actor_04']},
     'savee': {'val_speaker': ['DC'], 'test_speaker': ['JE']},# I concatenate val and train in model config for a 75% train/test split instead.
}

# Mel Spectrogram and MFCC creation Define constants 
 
FRAME_WIDTH = 512 # increase to 512 now 
NUM_SPECTROGRAM_BINS = 512 # 512 is recommended for speech (default is 2048 and suited for music)
NUM_MEL_BINS = 128
LOWER_EDGE_HERTZ = 80.0 # Human speech is not lower
UPPER_EDGE_HERTZ = 7600.0 # Higher is inaudbile to humans   
N_MFCC = 40
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Command-line arguments:
"""
which_dataset      : which dataset to process (incl. emodb, savee, iemocap, ravdess)
sample_rate       : sampling rate in Hz
sample_duration   : duration of each sample in seconds
z_score           : whether to z-score normalize at this preprocessing stage ('y' or 'n')
--output (optional): name for output directory name to avoid overwriting previous runs
ex.  python dataset_preprocess.py emodb 16000 4 y 
or ex. python dataset_preprocess.py savee 16000 3 n --output new_output_directory_name """

def parse_args():
    parser = argparse.ArgumentParser(
     description="Preprocess emotional speech datasets."
    )

    parser.add_argument(
        "which_dataset",
        choices=["emodb", "iemocap", "ravdess", "savee"],
        help="Dataset to preprocess"
    )

    parser.add_argument(
        "SAMPLE_RATE",
        type=int,
        help="Target sample rate (Hz)"
    )

    parser.add_argument(
        "SAMPLE_DURATION",
        type=int,
        help="Sample duration (seconds)"
    )

    parser.add_argument(
        "z_score",
        nargs="?",
        default="n",
        choices=["y", "n"],
        help="Apply z-score normalization (y/n). Default: n"
    )

    parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional name for output folder to avoid overwriting previous preprocessing runs"
    )    

    return parser.parse_args()


# Dataset preprocessing utulity functions
"""
To preprocess and train models I iterate through a CSV rather which contains direct paths to all audio samples
used in the study, rather than traversing through the original dataset folders, as this is simpler to implement 

The CSV files have been provided. 
However, The CSV files needs to be in same root folder as this script so the functions can find it. 
"""

def get_dataset_paths(which_dataset):   

    if which_dataset == "emodb":
        dataset_name = "EmoDB"        
        DATASET_PATH = os.path.join(root_dir, dataset_name)
        csv_path = os.path.join(root_dir, "csv", "emodb.csv")
        
    elif which_dataset == "iemocap":
        dataset_name = "IEMOCAP"         
        DATASET_PATH = os.path.join(root_dir, dataset_name, "IEMOCAP_full_release_withoutVideos","IEMOCAP_full_release")
        csv_path = os.path.join(root_dir, "csv", "iemocap.csv")
        
    elif which_dataset == "ravdess":
        dataset_name = "RAVDESS"        
        DATASET_PATH = os.path.join(root_dir, dataset_name)
        csv_path = os.path.join(root_dir, "csv", "ravdess.csv")
        
    elif which_dataset == "savee":
        dataset_name = "SAVEE"
        
        DATASET_PATH = os.path.join(root_dir, dataset_name, "AudioData")
        csv_path = os.path.join(root_dir, "csv", "savee.csv")
        
    else:
        raise ValueError("Incorrect dataset provided, options are: emodb, iemocap, ravdess, savee")
    
    return dataset_name, DATASET_PATH, csv_path


def listwavs(dataframe, SAMPLE_RATE, dataset_name, dataset_path):
     list_wavs = []
     for file in dataframe['file']:
          audio_file_path = audio_file_parser(file, dataset_name, dataset_path)
          print("audio_file_path = ", audio_file_path)
          x, _ = lr.load(audio_file_path, sr=SAMPLE_RATE)
          list_wavs.append(x)
     return list_wavs

def audio_file_parser(file, dataset_name, dataset_path):
          if dataset_name == "EmoDB":
               audio_file_path = os.path.join(dataset_path, os.path.normpath(file)) 
          elif dataset_name == "SAVEE":
               audio_file_path = os.path.join(dataset_path, os.path.normpath(file) + ".wav")
               
          elif dataset_name == "RAVDESS":
               audio_file_path = os.path.join(dataset_path, file[0:8], file[9:])              
          else: #IEMOCAP
               audio_file_path = os.path.join(dataset_path, os.path.normpath(file)) 
          
          return audio_file_path

def trim_wave(wave, SAMPLE_DURATION, SAMPLE_RATE):
     duration = int(SAMPLE_DURATION) * SAMPLE_RATE
     return wave[0:duration]

def pad_wave(wave, SAMPLE_DURATION, SAMPLE_RATE):
     duration = int(SAMPLE_DURATION) *  SAMPLE_RATE
     padding = int(duration - len(wave))
     if padding <= 0:
          return wave
     return np.pad(wave, (0, padding), 'constant')

def save_output(wave, filename, out_path, SAMPLE_RATE):
     subdirs = os.path.dirname(os.path.normpath(filename))            
     filename = os.path.splitext(os.path.basename(filename))[0] + ".wav"     
     save_dir = os.path.join(out_path, subdirs)      
     print("save_dir =", save_dir)     
     try:
          os.makedirs(save_dir, exist_ok=True)          
     except Exception as e:
          print("Error creating directories:", e)      
     full_path = os.path.join(save_dir, filename)
     sf.write(full_path, wave, SAMPLE_RATE, subtype='PCM_24')




# Main dataset preprocessing functions     
def data_split(which_dataset, df, DATASET_PATH):
          val_speaker = DATASET_SPEAKER_DEFAULTS[which_dataset]['val_speaker']
          test_speaker = DATASET_SPEAKER_DEFAULTS[which_dataset]['test_speaker']
          # Split based on speaker column
          df_val = df[df['speaker'].isin(val_speaker)].reset_index(drop=True)
          df_test = df[df['speaker'].isin(test_speaker)].reset_index(drop=True)
          df_train = df[~df['speaker'].isin(val_speaker)].reset_index(drop=True)
          df_train = df_train[~df_train['speaker'].isin(test_speaker)].reset_index(drop=True)

          if not os.path.exists(DATASET_PATH):
               os.makedirs(DATASET_PATH)
               

          df_train.to_csv(os.path.join(DATASET_PATH, 'train.csv'), index=False)
          print(df_train.isna().sum())  # total NaNs per column)
          df_val.to_csv(os.path.join(DATASET_PATH,'val.csv'), index=False)
          print(df_val.isna().sum())  # total NaNs per column)
          df_test.to_csv(os.path.join(DATASET_PATH,'test.csv'), index=False)
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
          
  
# Audio sample duration andsample rate adjustment, and optional z-score normalisation.
def norm_script(which_dataset, z_score, DATASET_PATH, dataset_name, SAMPLE_RATE, SAMPLE_DURATION, csv_path, out_path, output_arg, df):     

     # Z-score normalisation          
     """
     First compute the mean and std from the training data in order to fit
     # We compute from the training data to prevent data leakage and preserve the integrity of speaker independence
     # Note this is not ideal, as we are loading the entire training dataset into memory at once, but it is simpler to implement and the datasets are small enough that it should not cause memory issues. 
     # An alternative  implementation would compute the mean and std in a streaming fashion without loading everything at once, although this step would take longer to run in that case. """
     
     dataset_path = os.path.join(root_dir, DATASET_PATH)
     globalaudio = np.concatenate(listwavs(pd.read_csv(os.path.join(dataset_path,"train.csv")), SAMPLE_RATE, dataset_name, DATASET_PATH))
     mean = np.mean(globalaudio)
     std = np.std(globalaudio)
     print("Progress: global values for z-score normalisation calculated")


     for index, filepath in enumerate(pd.read_csv(csv_path)['file'].values):
          print("file =", filepath)

          audio_file = audio_file_parser(filepath,dataset_name, DATASET_PATH)  
          print("audio file = ", audio_file)
          y,sr = lr.load(audio_file,sr=SAMPLE_RATE)  

          #Normalise via zero mean and 1 unit variance if z-score at this stage is selected
          if z_score == 'y':
               y_norm =  (y - mean) / std
          else: 
               y_norm = y #Use the original sample instead of the normalised
          if not os.path.exists(out_path):
               os.makedirs(out_path)

          #Trimming or padding to a uniform duration, and writing the new sample to the output folder.     
          if lr.get_duration(y=y,sr=sr) > SAMPLE_DURATION:
               trimmed_wave = trim_wave(y_norm, SAMPLE_DURATION, SAMPLE_RATE)
               save_output(trimmed_wave,filepath, out_path, SAMPLE_RATE)               
          else: 
               padded_wave = pad_wave(y_norm, SAMPLE_DURATION, SAMPLE_RATE)
               save_output(padded_wave,filepath, out_path, SAMPLE_RATE)             
     
          # Writing out          
          df.loc[index, str(output_arg)] = os.path.join(out_path, str(filepath))        
          print(os.path.join(out_path, str(filepath)))   
          df.to_csv(which_dataset + output_arg + "_preprocessed.csv")

     print("Progress: Z-score normalisation and fixing of sample duration completed")
     print

   

# Deriving Mel Spectrograms and MFCCs from the normalised and duration-adjusted samples, and writing the paths to these new features to a CSV file     
def mel_mfcc(out_path, which_dataset, SAMPLE_RATE, df, output_arg):
     #Creating Mel and MFCCs. 
     
     # Creating the mel and mfcc directories in the output folder if they do not already exist.
     mel_path = os.path.join(out_path, "mel")
     mfcc_path = os.path.join(out_path, "mfccs")
     if not os.path.exists(mel_path):
          os.makedirs(mel_path)
     if not os.path.exists(mfcc_path):
          os.makedirs(mfcc_path)
     
     csv_path2 = os.path.join(which_dataset + output_arg + "_preprocessed.csv")
            
     file_column = output_arg       
     
     # Take the trimmed/padded sound files 
     for filepath in pd.read_csv(csv_path2)[file_column].values:

          print("file =", filepath)          
          if which_dataset == "savee":
               filepath_wav = os.path.normpath(filepath)+ ".wav"
          else:
               filepath_wav = os.path.normpath(filepath)
          
          samples, sample_rate = librosa.load(filepath_wav, sr=SAMPLE_RATE)

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
          
          # Get the relative path from the dataset root
          rel_path = os.path.relpath(filepath, start=out_path)  # e.g., 'wav/16b10Wb.wav'
          rel_dir = os.path.dirname(rel_path) 
          print("rel_dir is",rel_dir)
          filename = os.path.splitext(os.path.basename(filepath))[0]
                         
          mel_save_dir = os.path.join(mel_path, rel_dir)
          mfcc_save_dir = os.path.join(mfcc_path, rel_dir)
                    
          try:
               os.makedirs(mel_save_dir, exist_ok=True)
               os.makedirs(mfcc_save_dir, exist_ok=True)
          except Exception as e:
               print("Error creating directories:", e)
               continue  # skip this file

          mel_save_path = os.path.join(mel_save_dir, filename + ".npy") 
          np.save(mel_save_path, mel_sgram)
          print("saved mel spectrogram to", mel_save_path)
          mfcc_save_path = os.path.join(mfcc_save_dir, filename + ".npy")
          mfccs = librosa.feature.mfcc(S=mel_sgram,sr=SAMPLE_RATE,n_mfcc=N_MFCC)
          np.save(mfcc_save_path, mfccs)
          print("saved", mfccs, "MFCCs to ", mfcc_save_path)
          
          
     #Write path to the mel-spectrograms and mffcs for each utterance to the appropriate row in the CSV
     for index, filepath in enumerate(df['file'].values):   
          df.loc[index, 'mel_spectrogram'] = os.path.join(mel_path,str(filepath)[4:-4]+".npy")
          df.loc[index, 'MFCCs'] = os.path.join(mfcc_path, str(filepath)[4:-4]+".npy")

     df.to_csv(which_dataset + output_arg + "_preprocessed_with_mel_mfcc.csv")



# Process args and execute the main functions in order
def main():
    args = parse_args()

    which_dataset = args.which_dataset
    SAMPLE_RATE = args.SAMPLE_RATE
    SAMPLE_DURATION = args.SAMPLE_DURATION
    z_score = args.z_score

    print("Dataset selected:", which_dataset)

    # Sample rate warning logic
    if which_dataset == "ravdess":
        max_sr = 48000
        warning_msg = (
            "RAVDESS sample rate of source files are 48kHz, "
            "this script does not upsample. Continue? (y/n): "
        )
    else:
        max_sr = 16000
        warning_msg = (
            "Sample rate of source files are 16kHz, "
            "this script does not upsample. Continue? (y/n): "
        )

    if SAMPLE_RATE > max_sr:
        flag = input(warning_msg).lower()
        if flag != "y":
            print("Exiting.")
            sys.exit(1)

    print("Z-score normalisation:", z_score)

    # Get dataset paths
    dataset_name, DATASET_PATH, csv_path = get_dataset_paths(which_dataset)

    # Override if user supplied optional output directory argument
    output_dir = (
        args.output
        if args.output is not None       
    else ('norm_and_fixedduration' if z_score == 'y' else 'fixedduration')
    )
    out_path = os.path.normpath(os.path.join(dataset_name, output_dir))

    os.makedirs(out_path, exist_ok=True)

    df = pd.read_csv(csv_path)
    output_arg = output_dir

    
    # Run the main preprocessing functions in order
    data_split(which_dataset, df, DATASET_PATH)
    norm_script(which_dataset, z_score, DATASET_PATH, dataset_name, SAMPLE_RATE, SAMPLE_DURATION, csv_path, out_path, output_arg, df)
    mel_mfcc(out_path, which_dataset, SAMPLE_RATE, df, output_arg)

if __name__ == "__main__":
    main()
     
