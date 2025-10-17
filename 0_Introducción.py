import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(page_title="Fibrosis",
                   page_icon="🩺",
                   layout="wide")


# Título y descripción
st.title("Desarrollo de un modelo de regresión logística para la predicción de fibrosis hepática")

st.subheader("Introducción")
st.markdown("""
    <div style="text-align: justify; margin-bottom: 20px;">
        La fibrosis es el desarrollo en exceso de tejido en un órgano, lo que puede dificultar su función normal. Esta puede ser causada 
        por lesiones repetidas, exposición a contaminantes o como parte de una enfermedad genética.
    </div>
    """, unsafe_allow_html=True)
    
st.markdown("""
    <div style="text-align: justify; margin-bottom: 20px;">
        La Fibrosis hepática es la formación de una cantidad excesivamente grande de tejido cicatricial en el hígado, que puede llevar a la cirrosis 
        (reemplazo de tejido sano por tejido cicatricial) si la cicatrización es grave. Se produce cuando el hígado intenta reparar y reemplazar las 
        células dañadas.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: justify; margin-bottom: 20px;">
        El diagnóstico de la fibrosis se puede establecer a menudo a partir de pruebas de sangre o por métodos de diagnóstico por imágenes médicas,
        aunque a veces es necesaria una biopsia hepática, la cual es una prueba invasiva. Las técnicas de imágenes, aunque no son invasivas pueden ser 
        costosas o poco accesibles ya que se necesita un equipo especial. Por lo que el desarrollo de modelos predictivos basados en variables cuyo método de 
        obtención es no invasiva o mínimamente invasivas (como extracción de sangre) y cuya obtención es muchas veces rutinarias es muy prometedor para el 
        diagnostico temprano de la fibrosis hepática. Estos métodos ofrecen una alternativa confiable, no invasiva y debajo costo en comparación con métodos de 
        diagnóstico convencionales.
    </div>
    """, unsafe_allow_html=True)

st.subheader("Objetivo")
st.markdown("""
    <div style="text-align: justify; margin-bottom: 20px;">
        Desarrollar un modelo de regresión logística que permita estimar la probabilidad de presencia de fibrosis hepática.
    </div>
    """, unsafe_allow_html=True)