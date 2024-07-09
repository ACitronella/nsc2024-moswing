import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm
from moswing_inference.model_utils import model_predict, load_class_label, load_thres, load_model, load_model_setting, DURATION_EACH_DATAPOINT
from moswing_inference.sound_utils import DEFAULT_SR, load_sound, preprocess_file
from moswing_inference.plot_utils import plotly_overlap_prediction, binary_y

model_setting_dir = "moswing_inference/model_graveyard/m181_aecx" 
sound_dir = "moswing_inference/test_sound/SED-Test"

ms = load_model_setting(model_setting_dir)
model = load_model(ms)
thres = load_thres(ms)
class_labels = load_class_label(ms)
labels = list(class_labels.keys())
labels = sorted(labels, key=lambda val: class_labels[val])

file_list = st.file_uploader("Upload sound file (.wav only)", accept_multiple_files=True)
if file_list: 
    filtered_file = [f for f in file_list if str(f.name).lower().endswith(".wav")]
    file_names = (str(f.name) for f in filtered_file)
    sounds = (load_sound(f, DEFAULT_SR) for f in filtered_file)
    sounds = (preprocess_file(s) for s in sounds)
    predictions = (model_predict(model, s) for s in sounds)
 
    for file_name, pred in tqdm(zip(file_names, predictions), total=len(file_list)):
        # matplotlib version, this is not interactive plot
        # plt.figure(figsize=(18, 6))
        # plot_overlap_prediction(y_pred=pred, fig_title=file_name, duration_each_datapoint=DURATION_EACH_DATAPOINT, 
        #                         thresholds=thres, class_labels=class_labels)
        # st.pyplot(plt.gcf())
    
        fig = plotly_overlap_prediction(y_pred=pred, fig_title=file_name, duration_each_datapoint=DURATION_EACH_DATAPOINT, 
                                        thresholds=thres, class_labels=class_labels) 
        st.plotly_chart(fig)

        pred = pred.reshape(-1, len(class_labels))
        df = pd.DataFrame(pred, columns=labels)
        df["time(s)"] = DURATION_EACH_DATAPOINT*np.arange(pred.shape[0])
        s = df.to_csv()
        st.download_button("download raw prediction", data=s, file_name=f"{file_name}.rawpred.csv", mime="text/csv", key=f"{file_name}.rawpred.csv")

        pred = binary_y(pred, thres)
        df = pd.DataFrame(pred, columns=labels)
        df["time(s)"] = DURATION_EACH_DATAPOINT*np.arange(pred.shape[0])
        s = df.to_csv()
        st.download_button("download thresholded prediction", data=s, file_name=f"{file_name}.threpred.csv", mime="text/csv", key=f"{file_name}.threpred.csv")
