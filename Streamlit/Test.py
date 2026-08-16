import streamlit as st
import pickle
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(layout="wide")

with st.sidebar:
    st.title(" ")

BASE_DIR = Path(__file__).resolve().parent.parent

st.write("Project location:", BASE_DIR)
st.write("Model path:", BASE_DIR / "models" / "svm_model.pkl")
st.write("Model exists:", (BASE_DIR / "models" / "svm_model.pkl").exists())