# train_model.py
import argparse 
import sys
import os
print("cwd = ", os.getcwd())
print("sys.path = ", sys.path)

project_root = os.path.dirname(os.path.abspath(__file__))  # folder of train_model.py
if project_root not in sys.path:
    sys.path.append(project_root)
from prepare_tf_datasets import create_tf_datasets

#ex python train_model.py emodb mfcc 16000 4 32 fixedduration
# create_tf_datasets(emodb mfcc 16000 4 32 fixedduration)

# Argument handling
def parse_args():
    parser = argparse.ArgumentParser(description="Train SER model")

    parser.add_argument(
        "DATASET",
        type=str,
        choices=["emodb", "iemocap", "ravdess", "savee"],
        help="Dataset name"
    )

    parser.add_argument(
        "DATATYPE",
        type=str,
        choices=["wav", "mel", "mfcc"],
        help="Input data representation"
    )

    parser.add_argument(
        "SAMPLE_RATE",
        type=int,
        help="Audio sample rate"
    )

    parser.add_argument(
        "SAMPLE_DURATION",
        type=int,
        help="Audio duration in seconds"
    )

    parser.add_argument(
        "BATCH_SIZE",
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        "PREPROCESSED_ROOT_DIR",
        type=str,
        help="Preprocessed dataset subdirectory, this is the same as output directory used in dataset_preprocess.py"

    )

    args = parser.parse_args()
    return args

args = parse_args()
train_ds, val_ds, test_ds = create_tf_datasets(args.DATASET, args.DATATYPE, args.SAMPLE_RATE, args.SAMPLE_DURATION, args.BATCH_SIZE, args.PREPROCESSED_ROOT_DIR)
# model = build_model(args.new_arg_for_modelname)
# trained_model =  train_model(model)
# evaluate_model = model.evaluate() 
