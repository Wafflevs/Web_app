#pip install streamlit
import streamlit as st
import pandas as pd
import joblib
st.set_page_config(page_title="MLWebApp", layout="wide")


pipeline_path = "artifacts/preprocessor/preprocessor.pkl"
model_path = "artifacts/model/SVM.pkl"
encoder_path = "artifacts/preprocessor/label_encoder.pkl"


with open(pipeline_path, "rb") as file1:
    print(file1.read(100))
    

try:
    pipeline= joblib.load(pipeline_path)
    print("pipeline cargada")
    st.write("pipeline cargada")
except Exception as e:
    print(f"Error al cargar el pipeline {e}")
    



with open(model_path, "rb") as file2:
    print(file2.read(100))
try:
    model = joblib.load(model_path)
    print("modelo Cargado")
    st.write("modelo cargado")
except Exception as e:
    print("error al cargar modelo {e}")



with open(encoder_path, "rb") as file3:
    print(file3.read(100))
try:
    encoder = joblib.load(encoder_path)
    print("codificador cargado")
    st.write("encoder cargado")
except Exception as e:
    print("fallo al cargar el encoder {e}")


###################################################################

st.title("WebAPP de Machine Learning")
st.header("Ingreso de los datos")

col1, col2, col3 = st.columns(3)

with col1:

    battery_power = st.slider("Poder de la bateria (mAh)", min_value=500, max_value=2000, value=800)
    clock_speed = st.slider("velocidad del cpu", min_value=0.5, max_value=3.0)
    fc = st.slider("camara frontal (Mpx)", min_value=0, max_value=19, step=2)
    int_memory = st.slider("memoria interna (GB)", min_value=2, max_value=62, value=32)
    px_height = st.slider(
        "resolucion de la pantalla (altura en Px)", min_value=100,max_value=2000
    )


with col2:

    m_dep = st.slider("grososr del telefono", min_value=0.1, max_value=1.0)
    mobile_wt = st.slider("peso del telefono", min_value=100, max_value=1000)
    n_cores = st.slider("numero de nucleos", min_value=1, max_value=10)
    pc = st.slider("camara trasera MP", min_value=1, max_value=19)
    px_width = st.slider("Resolicion de la pantalla (ancho en PX)", min_value=100, max_value=2000)

with col3:

    ram = st.slider("Memoria RAM", min_value=256, max_value=4000)
    sc_h = st.slider("altura de la pantalla cm", min_value=10, max_value=12)
    sc_w = st.slider("Ancho de la pantalla", min_value= 0, max_value=18)
    talk_time = st.slider("Duracion de la bateria (Hrs)", min_value=2, max_value=20)