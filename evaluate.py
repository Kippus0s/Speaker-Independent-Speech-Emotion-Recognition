import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
from tensorflow import keras
from tensorflow.keras import Sequential, optimizers, layers
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2, l1, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.metrics import accuracy_score, classification_report

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_weight_history(model, history, filename,test_ds):
    #Saving weights and training history 
    os.makedirs("weights", exist_ok=True)
    model.save(os.path.join(root_dir, "weights", filename + "_weights.keras"))

    os.makedirs("training_history", exist_ok=True)
    with open((os.path.join(root_dir, "training_history", filename + "_history")), 'w') as f:
        json.dump(history.history, f)

   

def save_preds_and_true(model, filename, test_ds):

    os.makedirs("model_predictions", exist_ok=True)
    preds = model.predict(test_ds)
    np.save(str(filename) + "_preds", preds)
    np.save(os.path.join(root_dir, "model_predictions", filename + "_preds.npy"), preds)

    # Extract true labels for test set
    true_labels = []
    for _, labels in test_ds.unbatch():
        true_labels.append(labels.numpy())
    true_labels = np.array(true_labels)

    # Save for ensemble use
    os.makedirs("model_predictions", exist_ok=True)
    np.save(os.path.join("model_predictions", filename + "_true.npy"), true_labels)

def plot_cm(model, filename, test_ds, DATASET, LABEL_MAP):
        #Generate normalised confusion matrices
        os.makedirs("confusion_matrices", exist_ok=True)
        y_true = []
        y_pred = []
        for x, y in test_ds:
            preds = model.predict(x,verbose=0)  
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(y.numpy())
            
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        label_map = LABEL_MAP[DATASET]
        cm = confusion_matrix(y_true, y_pred, labels= range(len(label_map)))
        row_sums = cm.sum(axis=1, keepdims=True) 
        cmn = np.divide(cm.astype('float'), row_sums, where=row_sums!=0)
        
        
        
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cmn, display_labels=label_map)
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title(filename, fontsize=16)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        disp.plot(cmap="Blues",ax=ax)
        plt.savefig((os.path.join(root_dir, "confusion_matrices", filename + "_cm.png", )), dpi=300, bbox_inches='tight')

def display_class_report(model, test_ds):
    preds = model.predict(test_ds)
    classes = np.argmax(preds, axis=1)

    true_labels = []
    for _, labels in test_ds.unbatch():
        true_labels.append(labels.numpy())
    true_labels = np.array(true_labels)

    acc = accuracy_score(true_labels, classes)
    print(f'Accuracy: {acc:.4f}')

    print(classification_report(true_labels, classes, zero_division=0))
