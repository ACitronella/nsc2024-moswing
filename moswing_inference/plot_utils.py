import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pandas as pd

def binary_y(y, thresholds):
    tmp = np.zeros_like(y)
    for i in range(len(thresholds)):
        tmp[..., i] = (y[..., i] > thresholds[i]).astype(int)
    return tmp

def plot_overlap_prediction(y_true=None, y_pred=None, class_labels=None, fig_title=None, duration_each_datapoint=0.3, save_path=None, thresholds=None):
    assert class_labels is not None
    if y_true is not None:
        if thresholds is not None:
            y_true = binary_y(y_true, thresholds)
        reshaped_y_true = np.reshape(y_true, (-1, len(class_labels))) > 0.5 # training set
    if y_pred is not None:
        if thresholds is not None:
            y_pred = binary_y(y_pred, thresholds)
        reshaped_y_pred = np.reshape(y_pred, (-1, len(class_labels))) > 0.5 # training set

    if fig_title:
        plt.title(fig_title)
    class_name = class_labels.keys()
    n_classes = len(class_name)
    for idx, v in enumerate(class_name):
        if not (y_true is None):
            scaled_datapoints = (idx+1+0.2)/n_classes * reshaped_y_true[:, idx]
            plt.scatter(np.arange(reshaped_y_true.shape[0]), scaled_datapoints, s=100, alpha=0.2, color='black', marker="o")
        if not (y_pred is None):
            scaled_datapoints = ((idx+1-0.22)/n_classes * reshaped_y_pred[:, idx])
            plt.scatter(np.arange(reshaped_y_pred.shape[0]), scaled_datapoints, alpha=1, marker="o")
    if not (y_true is None):
        ticks = np.linspace(0, reshaped_y_true.shape[0], 11, endpoint=True)
    elif not (y_pred is None):
        ticks = np.linspace(0, reshaped_y_pred.shape[0], 11, endpoint=True)
    else:
        ticks = np.array([])
    plt.xticks(ticks, labels=[round(x, 2) for x in ticks*duration_each_datapoint])
    plt.xlabel("time (s)")
    plt.ylim(0.05, 1.05)
    plt.yticks(np.arange(0.1, 1.1, 1./n_classes), labels=class_name)
    plt.ylabel("species")
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    

def plotly_overlap_prediction(y_true=None, y_pred=None, class_labels=None, fig_title=None, duration_each_datapoint=0.3, save_path=None, thresholds=None):
    assert class_labels is not None
    # if y_true is not None:
    #     if thresholds is not None:
    #         y_true = binary_y(y_true, thresholds)
    #     reshaped_y_true = np.reshape(y_true, (-1, len(class_labels))) > 0.5 # training set
    if y_pred is not None:
        if thresholds is not None:
            y_pred = binary_y(y_pred, thresholds)
        reshaped_y_pred = np.reshape(y_pred, (-1, len(class_labels))) > 0.5 # training set
    anti_class_labels = {v : k for k, v in  class_labels.items()}

    argwhered_pred = np.argwhere(reshaped_y_pred)
    df = pd.DataFrame({"time (s)": argwhered_pred[:, 0],
                       "species": argwhered_pred[:, 1]})
    df["species"] = df["species"].apply(lambda cls: anti_class_labels[cls])
    return px.scatter(df, x="time (s)", y="species", color="species")
    