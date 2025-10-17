# Librerías
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

# Configuración de la página
st.set_page_config(page_title="EDA Fibrosis",
                   page_icon="📊",
                   layout="wide")

# Título
st.title("Análisis exploratorio de los dato")

# Datos
df = pd.read_csv('fibrosis.csv')

df['fibrosis'] = df['fibrosis'].map({0: 0, 1: 1, 2: 1, 3:1})
df['sexo'] = df['sexo'].map({1: 'M', 2: 'F'}).astype("category")

st.subheader("Primeras filas del conjunto de datos")
st.dataframe(df.head(10))

st.subheader("Descripción de las variables")
st.markdown("""
    <div style="text-align: justify;">
        En la siguiente tabla se puede ver una descripción de las variables utilizadas en este proyecto. Estas variables corresponden a variables 
        clínicas cuya obtención es rutinaria o poco invasiva, solo necesitando extracción de sangre, las cuales podrían ser útiles a la hora de determinar 
        la presencia o ausencia de fibrosis hepática.
    </div>
    """, unsafe_allow_html=True)

# Definir los datos
data = {
    "Variable": [
        "Glucemia", "Urea", "Creatinina", "Colesterol", "Triglicéridos", "GOT", "GPT", "GGT",
        "Fosfatasa alcalina", "Bilirrubina total", "Proteínas totales", "Albúmina", "Sodio", "Potasio",
        "Leucocitos", "Hematocrito", "Plaquetas", "Índice de Quick", "Fibrinógeno", "Sexo", "Fibrosis (objetivo)"
    ],
    "Descripción": [
        "Nivel de glucosa en sangre",
        "Producto de desecho renal",
        "Indicador de función renal",
        "Lípido total en sangre",
        "Tipo de grasa en sangre",
        "Enzima hepática",
        "Enzima hepática",
        "Enzima hepática",
        "Enzima hepática y ósea",
        "Producto de desecho del hígado",
        "Suma de proteínas plasmáticas",
        "Principal proteína plasmática",
        "Electrolito",
        "Electrolito",
        "Glóbulos blancos",
        "Proporción de glóbulos rojos",
        "Células de coagulación",
        "Tiempo de coagulación de la sangre usando reacción de Quick",
        "Proteína de la coagulación",
        "Masculino / Femenino",
        "Presencia / Ausencia (1 / 0)"
    ],
    "Tipo de variable": [
        "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica",
        "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Numérica",
        "Numérica", "Numérica", "Numérica", "Categórica (binaria)", "Categórica (binaria)"
    ]
}

# Tabla
df_variables = pd.DataFrame(data)
st.dataframe(df_variables, width='stretch')

# Separador
st.markdown("""
---
""")

# Mapeo de nombre de variables

variable_map = {
    "Glucemia": "glucemia",
    "Urea": "urea",
    "Creatinina": "creat",
    "Colesterol": "colester",
    "Triglicéridos": "trgl",
    "GOT": "got",
    "GPT": "gpt",
    "GGT": "ggt",
    "Fosfatasa alcalina": "fa",
    "Bilirrubina total": "biltotal",
    "Proteínas totales": "prottot",
    "Albúmina": "alb",
    "Sodio": "na",
    "Potasio": "k",
    "Leucocitos": "leucos",
    "Hematocrito": "hcto",
    "Plaquetas": "plaq",
    "Índice de Quick": "iq",
    "Fibrinógeno": "fibrin",
    "Sexo": "sexo"
}

# Sidebar
st.sidebar.header('Variables')
#variables = df.columns.drop(["fibrosis", "sexo"])
variable_visible = st.sidebar.selectbox('Por favor, seleccione la variable de interés:', list(variable_map.keys()))
variable_seleccionada = variable_map[variable_visible]

# Título variable seleccionada
st.markdown(f"## Análisis de la variable - {variable_visible}")

# Subconjunto de datos
valores = df[variable_seleccionada]
diagnostico = df['fibrosis']

if variable_visible == "Sexo":
    # Gráficos
    st.markdown(f"""
        <div style="text-align: justify;">
            A continuación, se presenta un diagrama de barras de la variable <strong>{variable_visible}</strong> y una tabla de sexo vs fibrosis. Podemos 
            observar que cuando hay ausencia de fibrosis el sexo femenino tiene mayor proporción con 61.7% (87) y el masculino con 38.3% (54), en cambio 
            cuando hay presencia de fibrosis la proporción de sexo masculino es mayor con 59.7% (108) y la femenina es menor con 40.3% (73).
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Diagrama de barras
    counts = df[variable_seleccionada].value_counts().reset_index()
    counts.columns = [variable_seleccionada, 'Frecuencia']
    counts['Porcentaje'] = (counts['Frecuencia'] / counts['Frecuencia'].sum()) * 100

    # Crear etiqueta personalizada con frecuencia y porcentaje
    counts['Etiqueta'] = counts['Frecuencia'].astype(str) + " (" + counts['Porcentaje'].round(1).astype(str) + "%)"
    
    with col1:
        fig_hist = px.bar(
            counts,
            x=variable_seleccionada,
            y='Frecuencia',
            text='Etiqueta',
            color=variable_seleccionada,
            title="Diagrama de Barras",
            color_discrete_sequence=['#1f77b4', '#e377c2']
        )
        
        fig_hist.update_traces(textposition='outside')  # Línea negra alrededor de las barras
        fig_hist.update_layout(xaxis_title=variable_visible, yaxis_title="Frecuencia", bargap=0.1, yaxis=dict(range=[0, counts['Frecuencia'].max() * 1.15]))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.markdown("<div style='height:120px'></div>", unsafe_allow_html=True) 
        st.markdown(
        """
        <p style="text-align: center; font-size:16px; font-weight:bold;">
            Distribución de fibrosis por sexo
        </p>
        """,
        unsafe_allow_html=True
        )

        # Crear tabla de frecuencias como DataFrame
        data_tabla = {
            ' ': ['Total', 'F', 'M'],
            'Total': ['322', '160 (49.7%)', '162 (50.3%)'],
            '0': ['141 (43.8%)', '87 (61.7%)', '54 (38.3%)'],
            '1': ['181 (56.2%)', '73 (40.3%)', '108 (59.7%)']
        }
        df_tabla = pd.DataFrame(data_tabla)

        # Mostrar tabla interactiva
        st.dataframe(df_tabla, use_container_width=True, hide_index=True)
        
    # Crear tabla de contingencia
    tabla_contingencia = pd.crosstab(df['sexo'], df['fibrosis'])
    chi2, p, dof, expected = stats.chi2_contingency(tabla_contingencia)
    
    # Resultados
    st.subheader("Asociación entre sexo y fibrosis")
    st.markdown(f"""
    <div style="text-align: justify;">
        Se realizó una prueba Chi-cuadrado para evaluar si hay asociación entre la variable <strong>{variable_visible}</strong> y <strong>fibrosis</strong>.
        El valor p obtenido fue <strong>{p:.3f}</strong>. {"Esto indica que hay evidencia estadísticamente significativa de asociación entre la varible sexo y fibrosis." if p < 0.05 else "No se encontraron diferencias significativas entre la varible sexo y fibrosis."}
    </div>
    """, unsafe_allow_html=True)
    
    st.write("**Frecuencias esperadas:**")
    st.dataframe(pd.DataFrame(expected, index=tabla_contingencia.index, columns=tabla_contingencia.columns), use_container_width=False)
        
else: 
    st.markdown(f"""
    <div style="text-align: justify;">
        A continuación se presenta una tabla con las estadísticas descriptivas de la variable <strong>{variable_visible}</strong>
        según la presencia o ausencia de fibrosis. También se presenta un histograma y un diagrama de caja y bigotes interactivos de 
        la variable Glucemia según la presencia o ausencia de fibrosis.
    </div>
    """, unsafe_allow_html=True)
    
    # Estadísticas descriptivas
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(valores.describe().to_frame().T.round(2), use_container_width=False)
    
    # Gráficos
    col1, col2 = st.columns(2)

    # Histograma
    with col1:
        fig_hist = px.histogram(
            df,
            x=variable_seleccionada,
            nbins=30,
            title="Histograma",
            color_discrete_sequence=["steelblue"]
        )
        
        fig_hist.update_traces(marker=dict(line=dict(color="black", width=1)))  # Línea negra alrededor de las barras
        fig_hist.update_layout(xaxis_title=variable_visible, yaxis_title="Frecuencia", bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Boxplot con colores personalizados
    with col2:
        fig_box = px.box(
            df,
            x='fibrosis',
            y=variable_seleccionada,
            color='fibrosis',
            title="Diagrama de Caja y Bigotes por Fibrosis",
            color_discrete_map={
                0 : 'steelblue',  # Azul para ausencia
                1 : 'firebrick'    # Rojo para presencia
            }
        )
        fig_box.update_layout( xaxis_title="Fibrosis (0 = Ausencia, 1 = Presencia)", yaxis_title=variable_visible)
        st.plotly_chart(fig_box, use_container_width=True)
        
    # Prueba de diferencias de medias
    
    # Filtrar los datos por diagnóstico
    grupo_ausencia = df[df['fibrosis'] == 0 ][variable_seleccionada].dropna()
    grupo_presencia = df[df['fibrosis'] == 1 ][variable_seleccionada].dropna()

    # Prueba t de Student para muestras independientes
    t_stat, p_valor = stats.ttest_ind(grupo_ausencia, grupo_presencia, equal_var=False)  # Welch’s t-test
    
    # Resultados
    st.subheader("Comparación de medias entre grupos de presencia y ausencia de fibrosis")
    st.markdown(f"""
    <div style="text-align: justify;">
        Se realizó una prueba t de Student para comparar las medias de la variable <strong>{variable_visible}</strong> entre los 
        grupos <strong>ausencia</strong> y <strong>presencia</strong> de fibrosis.
        El valor p obtenido fue <strong>{p_valor:.3f}</strong>. {"Esto indica una diferencia significativa entre los grupos." if p_valor < 0.05 else "No se encontraron diferencias significativas entre los grupos."}
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
---
""")

# Correlacion
st.subheader("Matriz de correlación entre las variables de estudio")
st.markdown("""
    <div style="text-align: justify;">
        En la en el mapa de calor de correlaciones se puede observar que GOT y GGT tienen una alta correlación (0.8), seguidas por correlaciones de valor 0.53 
        entre Urea y Creatinina, GGT y Fosfatasa alcalina, Leucocitos y Hematocritos. Las variables GGT y Bilirrubina total tienen una correlación de 0.43 
        mientras que Leucocitos y Fibrinógeno tienen una correlación de 0.40. Todas las correlaciones mencionadas anteriormente son positivas, las demás 
        correlaciones tienen un valor absoluto por debajo de 0.4.
    </div>
    """, unsafe_allow_html=True)

# Filtrado numérico y cálculo de correlación

mapa_nombres = {
    'glucemia': 'Glucemia',
    'urea': 'Urea',
    'creat': 'Creatinina',
    'colester': 'Colesterol',
    'trgl': 'Triglicéridos',
    'got': 'GOT',
    'gpt': 'GPT',
    'ggt': 'GGT',
    'fa': 'Fosfatasa alcalina',
    'biltotal': 'Bilirrubina total',
    'prottot': 'Proteínas totales',
    'alb': 'Albúmina',
    'na': 'Sodio',
    'k': 'Potasio',
    'leucos': 'Leucocitos',
    'hcto': 'Hematocrito',
    'plaq': 'Plaquetas',
    'iq': 'Índice de Quick',
    'fibrin': 'Fibrinógeno'
}

df_numericas = df.select_dtypes(include=[np.number]).drop(columns=['fibrosis'], errors='ignore')
corr = df_numericas.corr()
corr_renombrada = corr.rename(index=mapa_nombres, columns=mapa_nombres)

# Heatmap sin anotaciones
fig_corr = px.imshow(
    corr_renombrada,
    text_auto=".3f",                   # muestra valores numéricos en cada celda
    color_continuous_scale='RdBu_r',  # azul-rojo invertido (negativo a positivo) RdYlBu_r RdBu_r
    origin='lower',
    aspect='auto',
    title='Matriz de Correlación',
    zmin=-1,
    zmax=1
)

fig_corr.update_layout(
    title_x=0.5,
    width=800,
    height=600,
    margin=dict(l=50, r=50, t=80, b=50),
    coloraxis_colorbar=dict(title='Correlación')
)

st.plotly_chart(fig_corr, use_container_width=True)