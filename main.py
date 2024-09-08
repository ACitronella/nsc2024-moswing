import os
import streamlit as st
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from datetime import date, timedelta, datetime

from Data import add_location_to_data_instance, JsonDataStore
from GeoLocationFinder import find_latlong
from moswing_inference.model_utils import model_predict, load_class_label, load_thres, load_model, load_model_setting, DURATION_EACH_DATAPOINT
from moswing_inference.sound_utils import DEFAULT_SR, load_sound, preprocess_file
from moswing_inference.plot_utils import plotly_overlap_prediction, binary_y

DB_PATH = "data.json"
MODEL_SETTING_DIR = "moswing_inference/model_graveyard"

today = date.today()
today_timestamp = datetime.today().timestamp()
db = JsonDataStore(DB_PATH)

def adding_location_callback(file_name, file_id):
    file_name = copy.deepcopy(file_name)
    file_id = copy.deepcopy(file_id)
    def internal_fn():
        location_text = st.session_state.get(f"location_{file_id}", "")
        date_of_recording = st.session_state.get(f"date_{file_id}", "")
        print("retrived location:", location_text, "retrived date:", date_of_recording)
        if location_text != "" and date_of_recording != "":
            loc = find_latlong(location_text)
            if loc is not None:
                location_list.append(loc)
                loc["location_text"] = location_text
                add_location_to_data_instance(db, file_name, loc, date_of_recording)
                st.success(f"The location is saved {file_name}", icon="✅")
            else:
                print("cant find the location", location_text)
                st.warning(f"cant find the latlong of location {location_text}", icon="⚠️")

        else:
            print("some problem occurs here", location_text, date_of_recording)
            st.warning(f"some problem occurs here{location_text} {date_of_recording}", icon="⚠️")
    return internal_fn

def select_model_callback():
    selected_model = st.session_state.get("selected_model", None)
    if selected_model is None: 
        exit(1)
    st.session_state.model_setting_path = os.path.join(MODEL_SETTING_DIR, selected_model)

def file_uploader_callback():
    try:
        ms = load_model_setting(st.session_state.model_setting_path)
    except Exception as e:
        print(e)
        st.write("The selected model setting cannot be load properly.")
        exit(1)

    model = load_model(ms)
    thres = load_thres(ms)
    class_labels = load_class_label(ms)
    labels = list(class_labels.keys())
    labels = sorted(labels, key=lambda val: class_labels[val])

    existing_file_id = [file_id for (_, file_id, _, _, _) in st.session_state.plot_list]

    file_list = st.session_state.get("file_uploader")
    if file_list: 
        filtered_file = [f for f in file_list if str(f.name).lower().endswith(".wav") and str(f.file_id) not in existing_file_id]
        fild_ids = [str(f.file_id) for f in filtered_file]
        file_names = (str(f.name) for f in filtered_file)
        sounds = (load_sound(f, DEFAULT_SR) for f in filtered_file)
        sounds = (preprocess_file(s) for s in sounds)
        predictions = (model_predict(model, s) for s in sounds)
        
        for file_name, file_id, pred in tqdm(zip(file_names, fild_ids, predictions), total=len(filtered_file)):
            # matplotlib version, this is not interactive plot
            # plt.figure(figsize=(18, 6))
            # plot_overlap_prediction(y_pred=pred, fig_title=file_name, duration_each_datapoint=DURATION_EACH_DATAPOINT, 
            #                         thresholds=thres, class_labels=class_labels)
            # st.pyplot(plt.gcf())
        
            fig = plotly_overlap_prediction(y_pred=pred, fig_title=file_name, duration_each_datapoint=DURATION_EACH_DATAPOINT, 
                                            thresholds=thres, class_labels=class_labels) 

            pred = pred.reshape(-1, len(class_labels))
            df = pd.DataFrame(pred, columns=labels)
            df["time(s)"] = DURATION_EACH_DATAPOINT*np.arange(pred.shape[0])
            pred_csv = df.to_csv()

            pred = binary_y(pred, thres)
            df = pd.DataFrame(pred, columns=labels)
            df["time(s)"] = DURATION_EACH_DATAPOINT*np.arange(pred.shape[0])
            thresholded_pred_csv = df.to_csv()

            st.session_state.plot_list.append((file_name, file_id, fig, pred_csv, thresholded_pred_csv))

columns_to_be_shown = ["file_name", "location_text", "lat", "lon", "date"]
location_list:list[dict] = [v["location"] | {"timestamp": v["timestamp"], "file_name": k} for k, v in db.obj.items()]
is_location_empty = len(location_list) == 0
if not is_location_empty:
    location_df = pd.DataFrame(location_list, columns=["location_text", "lat", "lon", "timestamp"])
    location_df["date"] = location_df["timestamp"].apply(date.fromtimestamp)

st.set_page_config(layout="wide", page_title="MosWing: Inference")
if "plot_list" not in st.session_state:
    st.session_state.plot_list = [] # list[tuple[str, str, go.Figure, str, str]]
if "filtered_location_df" not in st.session_state:
    st.session_state.filtered_location_df = pd.DataFrame([], columns=columns_to_be_shown)

with st.container(border=True):
    st.title("Inference")
    st.selectbox("Select a SED model", os.listdir(MODEL_SETTING_DIR), key="selected_model")

    if "model_setting_path" not in st.session_state:
        st.session_state.model_setting_path = os.path.join(MODEL_SETTING_DIR, st.session_state.selected_model)

    st.file_uploader(
        "Upload sound file (.wav only)", accept_multiple_files=True, 
        on_change=file_uploader_callback, key="file_uploader"
    )

    for file_name, file_id, fig, pred_csv, thresholded_pred_csv in st.session_state.plot_list:
        with st.container(border=True):
            st.title(f"{file_name}'s prediction")
            st.plotly_chart(fig)
            _, col1, col2 = st.columns((6, 3, 3), gap="large", vertical_alignment="bottom")
            with col1:
                st.download_button("Download raw prediction", data=pred_csv, file_name=f"{file_name}-{st.session_state.get('selected_model', '')}-rawpred.csv", mime="text/csv", key=f"{file_id}-{st.session_state.get('selected_model', '')}-rawpred.csv", use_container_width=True)
            with col2:
                st.download_button("Download thresholded prediction", data=thresholded_pred_csv, file_name=f"{file_name}-{st.session_state.get('selected_model', '')}-csv", mime="text/csv", key=f"{file_id}--{st.session_state.get('selected_model', '')}-csv", use_container_width=True)

            col1, col2, col3 = st.columns((6, 3, 3), gap="large", vertical_alignment="bottom")
            with col1:
                location_text = db.obj.get(file_name, {}).get("location", {}).get("location_text", "")
                st.text_input(f"Location of {file_name}", key=f"location_{file_id}", value=location_text)
            with col2:
                date_stored = db.obj.get(file_name, {}).get("timestamp", None)
                date_value = date.fromtimestamp(date_stored) if date_stored is not None else None
                st.date_input(f"Date of recording {file_name}", key=f"date_{file_id}", value=date_value)
            with col3:
                st.button("Save location and date", key=f"{file_id}.addlocation", on_click=adding_location_callback(file_name, file_id), use_container_width=True)

with st.container(border=True):
    st.title(f"The location of mosquito records in a interval")
    st.write("The red dot indicates previous record of mosquito")
    st.date_input("Select date interval", value=(today - timedelta(days=30), today),
        format="YYYY/MM/DD", key="dates_on_map", 
    )    
    if len(st.session_state.dates_on_map) == 2:
        start_date, end_date = st.session_state.dates_on_map
        if not is_location_empty:
            st.session_state.filtered_location_df = location_df[location_df["date"].apply(lambda x: x >= start_date and x <= end_date)]

    st.map(st.session_state.filtered_location_df)
    st.dataframe(st.session_state.filtered_location_df[columns_to_be_shown], use_container_width=True)
