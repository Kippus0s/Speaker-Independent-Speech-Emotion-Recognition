import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
import audiomentations as au
from audiomentations import Compose, AirAbsorption, PitchShift, Shift, TimeStretch, AddGaussianNoise
import soundfile as sf
import matplotlib.pyplot as plt
import os
import time
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from prepare_tf_datasets import *
from evaluate import * 

#ex: python ensemble.py emodb_wav emodb_mel emodb_mfcc

"""trained on the same dataset exactly, they were trained on the same dataset but wav, mel and mfcc versions of it. but the correct label is the same as the test dataset isnt shuffled"""

def parse_args():
    parser = argparse.ArgumentParser(description="Late fusion ensemble from model predictions")
    
    parser.add_argument(
        "DATASET",
        type=str,
        choices=["emodb", "iemocap", "ravdess", "savee"],
        help="Dataset name"
    )

    parser.add_argument(
        "MODEL1",
        type=str,
        help="First model name (corresponding to *_preds.npy file)"
    )
    
    parser.add_argument(
        "MODEL2",
        type=str,
        help="Second model name (corresponding to *_preds.npy file)"
    )
    
    parser.add_argument(
        "MODEL3",
        type=str,
        help="Third model name (corresponding to *_preds.npy file)"
    )
    
    return parser.parse_args()

def main(args):    
    predictions_dir = os.path.join(os.getcwd(), "model_predictions")
    model_names = [args.MODEL1, args.MODEL2, args.MODEL3]
    DATASET = args.DATASET
    # Load predictions and true labels, and aggregate the probabilties
    model_preds = []
    for model_name in model_names:
        pred_file = os.path.join(predictions_dir, f"{model_name}_preds.npy")
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")
        preds = np.load(pred_file)
        print(pred_file, preds.shape)
        model_preds.append(preds)

    # Average predictions (decision level fusion)
    fused_preds = sum(model_preds) / len(model_preds)
    classes = np.argmax(fused_preds, axis=1)
    true_labels = np.load(os.path.join("model_predictions", args.MODEL3 + "_true.npy"))
    acc = accuracy_score(true_labels, classes)
    print(f'Accuracy: {acc:.4f}')

    #Return metrics
    print(classification_report(true_labels, classes, zero_division=0))
    
    #Plot CM    
    cm = confusion_matrix(true_labels, classes, labels= range(len(LABEL_MAP[DATASET])))
    # Normalise by number of true classes
    row_sums = cm.sum(axis=1, keepdims=True)    
    cmn = np.divide(cm.astype('float'), row_sums, where=row_sums!=0) 

    # Create the image and save to confusion matrices folder
    disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=LABEL_MAP[DATASET])
    fig, ax = plt.subplots(figsize=(10, 8))
    dataset = args.DATASET.upper()
    plt.title(dataset + " Ensemble", fontsize=16)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    disp.plot(cmap="Blues",ax=ax)
    plt.savefig((os.path.join(os.getcwd(), "confusion_matrices", dataset + "_Ensemble" + "_cm.png", )), dpi=300, bbox_inches='tight')
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)    

