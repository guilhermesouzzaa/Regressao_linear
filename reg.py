import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import scipy.stats as stats
import joblib
import streamlit as st
import pickle

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('train_energy_data.csv')

plt.style.use("ggplot")
lista_columns = ['Building Type','Square Footage','Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']
st.set_page_config(
    page_title="Previsão de gasto de energia elétrica",
    layout="wide"
    )

st.markdown(
    "<h1 style='text-align: center;'>Previsão de gasto de energia elétrica</h1>",
    unsafe_allow_html=True
)
lista_constr = ['Residential','Commercial', 'Industrial']
lista_semana = ['Dia útil', 'final de semana']
input_features = [st.selectbox("Tipo de construção", lista_constr),
                  st.number_input(
                    "Metros²",
                    min_value=1.0,
                    max_value=99999999999.0,
                    step=0.1), 
                    st.number_input("Digite número de ocupantes",
                                    min_value=1,
                                    max_value=999,
                                    step=1),
                    st.number_input("Eletros ligados",
                                    min_value=1,
                                    max_value=999,
                                    step=1),
                    st.number_input("Temperatura média",
                                    min_value=1,
                                    max_value=999,
                                    step=1),
                    st.selectbox("Dia útil ou final de semana?", lista_semana)]
if st.button("Processar"):
    input_df = pd.DataFrame([input_features], columns=lista_columns)
    input_df["Square Footage"] = input_df["Square Footage"] * 10.7639
    input_df['densidade_ocupacao'] = round(data['Square Footage']/data['Number of Occupants'])
    input_df['Building Type'] = input_df['Building Type'].map({'Residential': 0, 'Commercial': 1, 'Industrial':2 })
    input_df['Day of Week'] = input_df['Day of Week'].map({'Dia útil': 0, 'final de semana': 1})

    with open(
    "/home/guilherme-de-souza/Área de trabalho/minhas criações/regressão_linear/modelo_regressao.pkl",
    "rb") as f:
        modelo = joblib.load(f)

    previsao = modelo.predict(input_df)
    st.success(f"Previsão: {previsao[0]:.2f} kWh")


    
    