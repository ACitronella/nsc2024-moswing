import argparse
from typing import Iterator
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from .model_utils import load_model, load_model_setting, load_thres, load_class_label, DURATION_EACH_DATAPOINT, model_predict, ModelSetting
from .sound_utils import preprocess_file, load_sound, DEFAULT_SR
from .plot_utils import plot_overlap_prediction, binary_y

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_and_setting_dir", type=str)
    ap.add_argument("sound_dir", type=str)
    ap.add_argument("--output_dir", default="./out", type=str)
    return ap.parse_args()

def pipeline(model_setting_dir:str, sound_dir:str, output_dir:str) -> tuple[ModelSetting, Iterator[npt.NDArray], Iterator[str], list[str]] :
    ms = load_model_setting(model_setting_dir)
    model = load_model(ms)

    sound_file_names = os.listdir(sound_dir)
    sound_base_names = [os.path.basename(f) for f in sound_file_names]
    sound_paths = (os.path.join(sound_dir, f) for f in sound_file_names)
    sounds = (load_sound(p, DEFAULT_SR) for p in sound_paths)
    sounds = (preprocess_file(s) for s in sounds)

    predictions = (model_predict(model, s) for s in sounds)

    sound_output_dirs = (os.path.join(output_dir, f) for f in sound_base_names)

    return ms, predictions, sound_output_dirs, sound_base_names

def main(model_setting_dir:str, sound_dir:str, output_dir:str):

    os.makedirs(output_dir, exist_ok=True)

    ms, predictions, sound_output_dirs, sound_base_names = pipeline(model_setting_dir, sound_dir, output_dir)
    thres = load_thres(ms)
    class_labels = load_class_label(ms)
    labels = list(class_labels.keys())
    labels = sorted(labels, key=lambda val: class_labels[val])

    for basename, outname, pred in tqdm(zip(sound_base_names, sound_output_dirs, predictions), total=len(sound_base_names)):
        os.makedirs(outname, exist_ok=True)
        plt.figure(figsize=(18, 6))
        plot_overlap_prediction(y_pred=pred, fig_title=basename, duration_each_datapoint=DURATION_EACH_DATAPOINT, 
                                save_path=os.path.join(outname, "plot_prediction.jpg"), thresholds=thres, class_labels=class_labels) 
        plt.close("all")

        pred = pred.reshape(-1, len(class_labels))
        
        df = pd.DataFrame(pred, columns=labels)
        df["time(s)"] = DURATION_EACH_DATAPOINT*np.arange(pred.shape[0])
        df.to_csv(os.path.join(outname, "raw_predict.csv"))
        
        pred = binary_y(pred, thres)
        df = pd.DataFrame(pred, columns=labels)
        df["time(s)"] = DURATION_EACH_DATAPOINT*np.arange(pred.shape[0])
        df.to_csv(os.path.join(outname, "thres_predict.csv"))


if __name__ == "__main__":
    args = parse_args()
    model_and_setting_dir = args.model_and_setting_dir
    sound_dir = args.sound_dir
    output_dir = args.output_dir

    main(model_and_setting_dir, sound_dir, output_dir)

    