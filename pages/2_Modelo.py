# Librerías
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Configuración de la página
st.set_page_config(page_title="Fibrosis",
                   page_icon="🩺",
                   layout="wide")

# Título
st.title("Modelo de Regresión logística")

# Se carga el modelo
modelo = joblib.load('modelo_fibrosis.pkl')

# Datos
df = pd.read_csv('test_fibrosis.csv')

y_test = df['fibrosis']
X_test = df.drop(columns=['fibrosis'])

# Predicciones
y_prob = modelo.predict(X_test)

# Punto de corte óptimo usando el índice de Youden
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

# Sidebar para punto de corte
st.sidebar.header('Punto de corte')
threshold = st.sidebar.slider('Seleccione el punto de corte:',   
    min_value=0.0,
    max_value=1.0,
    value=float(optimal_threshold),
    step=0.01)

# Predicciones - probabilidad clase positiva 
y_pred = (y_prob >= threshold).astype(int)

# Calcular matriz de confusión para obtener especificidad
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Calcular métricas
accuracy = accuracy_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)   # TPR
specificity = tn / (tn + fp)                 # TNR
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Datos de la tabla
data = {
    "Coeficiente": [7.7371, -1.4339, -0.0031, 0.0129, 0.0032, 0.5169, -1.5430, -0.0002, 0.0713, '7.632e-06', -0.0789],
    "Error Estándar": [3.444, 0.323, 0.002, 0.004, 0.002, 0.288, 0.493, '0.000', 0.040, '4.14e-06', 0.027],
    "Z": [2.246, -4.433, -1.666, 3.079, 1.675, 1.797, -3.129, -1.822, 1.775, 1.843, -2.959],
    "P-Valor": [0.025, '0.000', 0.096, 0.002, 0.094, 0.072, 0.002, 0.068, 0.076, 0.065, 0.003],
}

# Índice (nombre de las variables)
index = [
    "Intercepto",
    "Sexo (Femenino)",
    "Triglicéridos",
    "GOT",
    "GGT",
    "Proteínas Totales",
    "Albúmina",
    "Leucocitos",
    "Hematocrito",
    "Plaquetas",
    "Índice de Quick"
]

# Crear DataFrame
df_resultados = pd.DataFrame(data, index=index)

# Título
st.subheader("Resultados del modelo de regresión logística")
st.markdown("""
    <div style="text-align: justify;">
    En la siguiente tabla se muestra el resumen del modelo, las variables sexo, triglicéridos, GOT, GGT, Proteínas totales, albúmina, leucocitos, 
    hematocrito, plaquetas e índice de Quick resultaron significativas. Al aumentar los valores de las variables triglicéridos, albúmina, leucocitos, 
    e índice de Quick se disminuyen la probabilidad de presencia de fibrosis, mientras que las otras variables numéricas aumentan esta probabilidad al 
    aumentar su valor.
    </div>
    """, unsafe_allow_html=True)
st.dataframe(df_resultados, use_container_width=False)

st.markdown("""
---
""")

# Crear tabla con métricas
metricas = pd.DataFrame({
    "Métrica": ["Exactitud", "Sensibilidad", "Especificidad", "AUC"],
    "Valor": [accuracy, sensitivity, specificity, auc]
})

# Formatear valores
metricas["Valor"] = metricas["Valor"].apply(lambda x: f"{x:.3f}")

# Mostrar en Streamlit
st.subheader("Métricas de desempeño del modelo")
st.markdown("""
    <div style="text-align: justify;">
    De acuerdo con la métrica de Exactitud del modelo, se obtiene que el modelo clasifica de manera correcta un 71.9% de las observaciones. Según la 
    sensibilidad, el modelo tiene una probabilidad del 72.2% de predecir correctamente a los pacientes con fibrosis. De acuerdo con la especificidad,
    el modelo predice a los pacientes con ausencia de fibrosis correctamente un 71.4% de las veces.
    </div>
    """, unsafe_allow_html=True)
st.dataframe(metricas, use_container_width=False)

st.subheader("Matriz de confusión y Curva ROC del modelo")
st.markdown("""
    <div style="text-align: justify;">
    A continuación, se muestra la curva de ROC y la matriz de confusión del modelo. El área bajo la curva de ROC fue de 0.738. El punto de corte que 
    maximiza el índice de Youden es 0.485, la matriz de confusión del modelo se realizó usando este punto de corte.
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Crear heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title('Matriz de Confusión')
    ax.xaxis.set_ticklabels(["Ausencia", "Presencia"])
    ax.yaxis.set_ticklabels(["Ausencia", "Presencia"])
    st.pyplot(fig, use_container_width=False)

with col2:
    # Curva ROC 
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('FPR (1 - Especificidad)')
    ax.set_ylabel('TPR (Sensibilidad)')
    ax.set_title('Curva ROC')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)

st.markdown("""
---
""")

# Crear inputs con valores por defecto en la media

st.subheader("Predicción de presencia o ausencia de fibrosis")
st.markdown("""
    <div style="text-align: justify;">
    Con el modelo de regresión logística podemos estimar la probabilidad de que un paciente presente o no fibrosis. Ingrese los valores en los 
    siguientes campos para realizar la predicción.
    </div>
    """, unsafe_allow_html=True)

df = pd.read_csv('fibrosis.csv')

df['fibrosis'] = df['fibrosis'].map({0: 0, 1: 1, 2: 1, 3:1})
df['sexo'] = df['sexo'].map({1: 'M', 2: 'F'}).astype("category")

medias = df.drop(columns=["sexo", "fibrosis"]).mean()

variables = {
    "trgl": "Triglicéridos",
    "got": "GOT",
    "ggt": "GGT",
    "prottot": "Proteínas totales",
    "alb": "Albúmina",
    "leucos": "Leucocitos",
    "hcto": "Hematocrito",
    "plaq": "Plaquetas",
    "iq": "Índice de Quick",
    "sexo": "Sexo"
}

num_vars = ["trgl", "got", "ggt", "prottot", "alb", "leucos", "hcto", "plaq", "iq"]

# Inputs numéricos en filas de 3 columnas
input_data = {}
cols_per_row = 3

for i in range(0, len(num_vars), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col_name in enumerate(num_vars[i:i+cols_per_row]):
        with cols[j]:
            input_data[col_name] = st.number_input(
                f"{variables[col_name]}",
                value=float(medias[col_name]),
                format="%.3f"
            )

# Input categórico
sexo = st.selectbox("Sexo", options=["Masculino", "Femenino"])
input_data['sexo'] = sexo

df_input = pd.DataFrame([input_data])

if st.button("Predecir"):
    prob = modelo.predict(df_input)[0]
    pred = int(prob >= threshold)

    # Asignar color según el nivel de probabilidad
    if prob >= threshold:
        color = "red"
    else:
        color = "green"

    # Mostrar resultado con colores
    st.subheader("Resultado de la Predicción")
    if pred == 1:
        st.error(f"Presencia de Fibrosis — Probabilidad: {prob:.2%}")
    else:
        st.success(f"Ausencia de Fibrosis — Probabilidad: {prob:.2%}")