# -*- coding: utf-8 -*-
import os, sys
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import subprocess
import json
import os
import plotly.express as px
import altair as alt
import time
import itertools
from pathlib import Path

# Debe ser la primera llamada de Streamlit
st.set_page_config(layout="wide")

def llamada_graficar(results, df_matriz_final):

    step = 10
    high = 300

    from Graficos import HEATMAP_PV_WIND, GRAPH_EBITDA_DISP, HEATMAP_PV_BESS, GRAPH_BESS

    if results["solar_factor"] == 1 and results["wind_factor"] == 1 and results["bess_factor"] == 1:    
        GRAPH_EBITDA_DISP(df_matriz_final)

    if results["solar_factor"] == 1 and results["wind_factor"] == 1 and results["bess_factor"] == 0: 
        HEATMAP_PV_WIND(df_matriz_final, step, results["ppa_mw"], high)  

    if results["solar_factor"] == 1 and results["wind_factor"] == 0 and results["bess_factor"] == 1: 
        HEATMAP_PV_BESS(df_matriz_final, step, results["ppa_mw"], high) 

    if results["solar_factor"] == 0 and results["wind_factor"] == 0 and results["bess_factor"] == 1: 
        GRAPH_BESS(df_matriz_final, step, results["ppa_mw"]) 


    status_text.markdown(
            "Estado Simulacion: <span style='color: green; font-weight: bold;'>Calculo Finalizado!</span>",
            unsafe_allow_html=True
    )


#------------------------------- LEER -------------------------------------#

#Obtener inputs de Excel
from Calculo_modelo import leer_ppa, tabla_rer

def busca_ruta():

    rutas_base = [
        "C:/Users/fnovoa/SOLARPACK/VENTAS ENERGIA - LATAM/0 LATAM Coordination/Product Structuring/Python/Model PortOptm",
        "C:/Users/IonHualdeIriondo/OneDrive - Zelestra Corporaci\u00f3n S.A.U/LATAM/0 LATAM Coordination/Product Structuring/Python/Model PortOptm",
        "C:/Users/OscarMorales/Zelestra Corporaci\u00f3n S.A.U/VENTAS ENERGIA - LATAM/0 LATAM Coordination/Product Structuring/Python/Model PortOptm"
    ]

    ruta_base_ok = None
    ruta_ppa = None
    ruta_modelinput = None

    for ruta in rutas_base:
        try:
            base = Path(ruta)

            ppa_tmp = base / "Input_PPA_MW.xlsx"
            modelinput_tmp = base / "Model_Input.xlsx"

            if ppa_tmp.exists() and modelinput_tmp.exists():
                ruta_base_ok = base
                ruta_ppa = ppa_tmp
                ruta_modelinput = modelinput_tmp
                #print(f"Ruta valida encontrada:\n{ruta_base_ok}")
                break

        except Exception:
            continue

    if ruta_base_ok is None:
        raise FileNotFoundError(
            "No se encontraron Input_PPA_MW.xlsx y Model_Input.xlsx en ninguna ruta"
        )

    return ruta_modelinput, ruta_ppa

ruta_modelinput, ruta_ppa = busca_ruta()

df_proyectos, df_generaciones = tabla_rer(ruta_modelinput)
df_ppa_long = leer_ppa(ruta_ppa)

#Boton y estado simulacion
ejecutar = st.sidebar.button("Ejecutar modelo")
status_text = st.empty()

#------------------------------------------------ INICIAR API -----------------------------------------------#

def Inicio_API():
    
    #------------------------------- MAPA -------------------------------------#

    nodos = ["Crucero", "Cardones", "Pan de Azucar", "Polpaico", "Charrua", "Puerto Montt"]
    nodos_bess = df_proyectos["Barra"].unique().tolist()
    nombres_pv = df_proyectos.loc[df_proyectos["Tech"] == "PV", "Projecto"].tolist()
    nombres_wind = df_proyectos.loc[df_proyectos["Tech"] == "WIND", "Projecto"].tolist()

    # --- Datos de ubicaciones para el mapa (solo PV y WIND) ---
    data_ubicaciones = (
        df_proyectos
        .loc[:, ["Tech", "Projecto", "LAT", "LON"]]
        .rename(columns={
            "Tech": "tipo",
            "Projecto": "nombre",
            "LAT": "lat",
            "LON": "lon"
        })
        .reset_index(drop=True)
    )

    # --- Colores por tipo ---
    def color_picker(tipo):
        return [255, 140, 0, 160] if tipo == "PV" else [0, 112, 255, 160]

    data_ubicaciones['color'] = data_ubicaciones['tipo'].apply(color_picker)

    # --- Mostrar mapa en la barra lateral ---
    st.sidebar.subheader("Mapa Desarrollo")
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data_ubicaciones,
        get_position='[lon, lat]',
        get_fill_color='color',
        get_radius=30000,
        pickable=True,
    )

    view_state = pdk.ViewState(latitude=-33.5, longitude=-70.8, zoom=5, pitch=0)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{tipo}: {nombre}"})
    st.sidebar.pydeck_chart(r)

    mapa_pv = (
        df_proyectos
        .loc[df_proyectos["Tech"].str.upper() == "PV", ["Projecto", "Barra"]]
        .drop_duplicates()
        .set_index("Projecto")["Barra"]
        .to_dict()
    )

    mapa_wind = (
        df_proyectos
        .loc[df_proyectos["Tech"].str.upper() == "WIND", ["Projecto", "Barra"]]
        .drop_duplicates()
        .set_index("Projecto")["Barra"]
        .to_dict()
    )

    #------------------------------- TECNOLOGIAS -------------------------------------#

    solar_factor = st.sidebar.checkbox("Activar PV")
    wind_factor = st.sidebar.checkbox("Activar WIND")
    bess_factor = st.sidebar.checkbox("Activar BESS")

    results = {
        "anio_inicio": 0,
        "anios_analisis": 0,
        "nombre_pv": "",
        "nodo_pv": "",
        "coste_pv": 0,
        "nombre_wind": "",
        "nodo_wind": "",
        "coste_wind": 0,
        "nombre_bess": "",
        "nodo_bess": "",
        "coste_bess": {
            "5h": 0,
            "4h": 0,
            "3h": 0,
            "2h": 0,
            "1h": 0
        }
    }

    if solar_factor:
        with st.sidebar.expander("**Configuracion Solar**", expanded=True):

            nombre_pv = st.selectbox("Nombre planta PV", nombres_pv)
            nodo_pv = mapa_pv.get(nombre_pv, "")
            st.text_input("Nodo PV", value=nodo_pv, disabled=True)
            coste_pv = st.number_input("Coste PV [$/MWh]", value=0.0)
            results.update({
                    "nombre_pv": nombre_pv,
                    "nodo_pv": nodo_pv,
                    "coste_pv": coste_pv
            })

    if wind_factor:
        with st.sidebar.expander("**Configuracion Eolico**", expanded=True):

            nombre_wind = st.selectbox("Nombre planta WIND", nombres_wind)
            nodo_wind = mapa_wind.get(nombre_wind, "")
            st.text_input("Nodo WIND", value=nodo_wind, disabled=True)
            coste_wind = st.number_input("Coste WIND [$/MWh]", value=0)
            results.update({
                    "nombre_wind": nombre_wind,
                    "nodo_wind": nodo_wind,
                    "coste_wind": coste_wind
            })

    if bess_factor:
        with st.sidebar.expander("**Configuracion BESS**", expanded=True):

            nodo_bess = st.selectbox("Nodo BESS", nodos_bess)    
            coste_bess = {}
            for h in range(5, 0, -1):  # De 5h a 1h
                coste_bess[f"{h}h"] = st.number_input(f"Coste BESS {h}h [$/MWh]", value=0, key=f"bess_{h}")
    
            results.update({
                    "nodo_bess": nodo_bess,
                    "coste_bess": coste_bess
            })

    with st.sidebar.expander("**Seleccion de PPA**", expanded=True):
            
        col1, col2 = st.columns(2)
        with col1:
        #Exprtar archivo PPA
            if st.button("Exportar"):
                 os.startfile(ruta_ppa)
        with col2:
            if st.button("Importar"):
                df_ppa_long = leer_ppa(ruta_ppa)

        #ppa_mw = st.number_input("PPA [MW]", value=50.0)
        ppa_price = st.number_input("PPA Price [$]", value=55.0)
        #estructura_ppa = st.selectbox("Estructura PPA", ["ABC", "AC", "B"])
        nodo_retiro = st.selectbox("Nodo de Retiro", nodos)


    results.update({
                #"ppa_mw": ppa_mw,
                "ppa_price": ppa_price,
                "nodo_retiro": nodo_retiro,
                #"estructura_ppa": estructura_ppa,
                "solar_factor": int(solar_factor),
                "wind_factor": int(wind_factor),
                "bess_factor": int(bess_factor)
    })

    with st.sidebar.expander("**Configuracion de Simulacion**", expanded=True):
        
        anio_inicio = st.selectbox("Selecciona Inicio",options=list(range(2027, 2035)))
        anios_analisis = st.selectbox("Selecciona Anios de Analisis",options=list(range(15, 26)))
        n_hydro = st.selectbox("Numero de hidrologias",options=list(range(1, 121)),index=119)
        wacc = st.number_input("WACC [%]", value=10.0)
         
    results.update({
        "anio_inicio": anio_inicio,
        "anios_analisis": anios_analisis,
        "wacc": wacc,
        "n_hydro": n_hydro
    })

    return results

results = Inicio_API()


tabx, tab0, tab1, tab2, tab3, tab4 = st.tabs(["Inputs", "Inyeccion Planta/BESS", "Combinaciones", "Margen Planta", "Margen Contrato", "Margen Proyecto"])



#--------------------------------------------- MOSTRAR INPUTS ---------------------------------------------#

def Inputs():

    df_tabla = df_ppa_long.pivot(
        index="HORA",
        columns="MES",
        values="PPA_MW"
    ).sort_index()

    
    df_long = (
        df_tabla
        .reset_index()
        .melt(id_vars="HORA", var_name="MES", value_name="PPA_MW")
    )

    base = alt.Chart(df_long).encode(
        x=alt.X(
            "MES:O",
            title="Mes",
            sort=list(range(1, 13)),   # todos los meses
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "HORA:O",
            title="Hora",
            sort=list(range(24))       # todas las horas
        )
    )

    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "PPA_MW:Q",
            title="PPA MW",
            scale=alt.Scale(
                scheme="redyellowgreen"   # rojo -> amarillo -> verde
            )
        )
    )

    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("PPA_MW:Q", format=".0f"),
        color=alt.value("black")
    )

    chart = (heatmap + text).properties(
        width=750,
        height=450
    )

    st.subheader("PPA")
    st.altair_chart(chart, width='stretch')




with tabx:
    Inputs()

#-------------------------------- CALCULAR CAPTURADOS E INGRESOS UNITARIOS -----------------------------------#

def grafica1(df_stats_mes, df_gen_mensual, title_0):

            # Merge
            df_plot = df_stats_mes.merge(
                df_gen_mensual,
                on=["ANO", "MES"],
                how="left"
            )

            # Fecha
            df_plot["FECHA"] = pd.to_datetime(
                df_plot["ANO"].astype(str) + "-" +
                df_plot["MES"].astype(str).str.zfill(2) + "-01"
            )

            # Base comun
            base = alt.Chart(df_plot).encode(
                x=alt.X("FECHA:T", title="Fecha")
            )

            # ----- CHART IZQUIERDO (MISMA ESCALA Y) -----

            area = base.mark_area(opacity=0.25).encode(
                y=alt.Y(
                    "P5:Q",
                    title="Capturado [$/MWh]",
                    stack=None,
                    scale=alt.Scale(zero=False)
                ),
                y2=alt.Y2("P95:Q")
            )

            line_ing = base.mark_line(size=2).encode(
                y=alt.Y("PROM:Q")
            )

            left = alt.layer(area, line_ing)

            # ----- CHART DERECHO (OTRA ESCALA Y) -----

            right = base.mark_line(
                size=2,
                color="red"
            ).encode(
                y=alt.Y(
                    "GENERACION:Q",
                    title="Generacion [MWh]",
                    axis=alt.Axis(orient="right"),
                    scale=alt.Scale(zero=False)
                )
            )

            # ----- LEYENDA FANTASMA -----

            legend = alt.Chart(
                pd.DataFrame({
                    "Serie": ["Capturado", "Generacion"],
                    "Valor": [0, 0]
                })
            ).mark_line().encode(
                y=alt.Y("Valor:Q", axis=None),
                color=alt.Color(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=["Capturado", "Generacion"],
                        range=["#1f77b4", "red"]
                    ),
                    legend=alt.Legend(title="")
                )
            )

            # ----- CHART FINAL -----

            chart = alt.layer(
                left,
                right,
                legend
            ).resolve_scale(
                y="independent"
            ).properties(
                width=700,
                height=400,
                title=title_0
            )

            st.altair_chart(chart, width="stretch")

def Calculo_IngresoUnitario(results):   

        from Calculo_modelo import generacion, cmg, Revenues_Generador, Energia_BESS, Revenue_BESS


        if "nodo_pv_anterior" not in st.session_state:

            df_ingresos_gen = []
            df_ingresos_gen2 = []
            resultados_BESS = []

            df_gen_mensual = []
            df_gen_mensual2 = []
            df_gen_mensual3 = []

            st.session_state.nodo_pv_anterior = ""
            st.session_state.nodo_wind_anterior = ""
            st.session_state.nodo_bess_anterior = ""

            st.session_state.nombre_pv_anterior = ""
            st.session_state.nombre_wind_anterior= ""

            st.session_state.anio_inicio_anterior = ""
            st.session_state.anios_analisis_anterior= ""

            st.session_state.df_ingresos_gen = df_ingresos_gen
            st.session_state.df_gen_mensual = df_gen_mensual

            st.session_state.df_ingresos_gen2 = df_ingresos_gen2
            st.session_state.df_gen_mensual2 = df_gen_mensual2

            st.session_state.resultados_BESS = resultados_BESS
            st.session_state.df_gen_mensual3 = df_gen_mensual3

        if results["solar_factor"] == 0:
            st.session_state.mostrar_graficas_pv = False
        if results["wind_factor"] == 0:
            st.session_state.mostrar_graficas_wind = False
        if results["bess_factor"] == 0:
            st.session_state.mostrar_graficas_bess = False

        if results["solar_factor"] == 1 and (results["nombre_pv"] != st.session_state.get("nombre_pv_anterior") or results["nodo_pv"] != st.session_state.get("nodo_pv_anterior") or results["anios_analisis"] != st.session_state.get("anios_analisis_anterior") or results["anio_inicio"] != st.session_state.get("anio_inicio_anterior")):
                
                st.session_state.nombre_pv_anterior = results["nombre_pv"]
                st.session_state.nodo_pv_anterior = results["nodo_pv"]
                st.session_state.mostrar_graficas_pv = True

                status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Ingresos PV...</span>",
                    unsafe_allow_html=True
                )
                time.sleep(1)

                #Generacion
                df_gen_tot, df_gen_mensual, df_gen_diario = generacion(results["nombre_pv"], results["anios_analisis"], results["anio_inicio"], df_proyectos, df_generaciones)

                # Coste marginal con las hidrologias
                df_prec = cmg(ruta_modelinput, results["nodo_pv"], results["anio_inicio"], results["anios_analisis"])

                #Ingresos del generador por hidrologia
                df_ingresos_gen, df_stats_mes = Revenues_Generador(df_prec, df_gen_tot, df_gen_mensual)

                st.session_state.df_ingresos_gen = df_ingresos_gen
                st.session_state.df_gen_mensual = df_gen_mensual
                st.session_state.df_stats_mes = df_stats_mes
                st.session_state.df_gen_diario = df_gen_diario

        if st.session_state.mostrar_graficas_pv:

            df_stats_mes = st.session_state.df_stats_mes
            df_gen_diario = st.session_state.df_gen_diario

            #Graficar
            grafica1(df_stats_mes, df_gen_diario, "Capturado Solar y Generacion Unitaria Diario")

        if results["wind_factor"] == 1 and (results["nombre_wind"] != st.session_state.get("nombre_wind_anterior") or results["nodo_wind"] != st.session_state.get("nodo_wind_anterior") or results["anios_analisis"] != st.session_state.get("anios_analisis_anterior") or results["anio_inicio"] != st.session_state.get("anio_inicio_anterior")):

                st.session_state.nombre_wind_anterior = results["nombre_wind"]
                st.session_state.nodo_wind_anterior = results["nodo_wind"]
                st.session_state.mostrar_graficas_wind = True

                status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Ingresos WIND...</span>",
                    unsafe_allow_html=True
                )
                time.sleep(1)

                #Generacion
                df_gen_tot2, df_gen_mensual2, df_gen_diario2 = generacion(results["nombre_wind"], results["anios_analisis"], results["anio_inicio"], df_proyectos, df_generaciones)

                # Coste marginal con las hidrologias
                df_prec2 = cmg(ruta_modelinput, results["nodo_wind"], results["anio_inicio"], results["anios_analisis"])

                #Ingresos del generador por hidrologia
                df_ingresos_gen2, df_stats_mes2 = Revenues_Generador(df_prec2, df_gen_tot2, df_gen_mensual2)

                st.session_state.df_ingresos_gen2 = df_ingresos_gen2
                st.session_state.df_gen_mensual2 = df_gen_mensual2
                st.session_state.df_stats_mes2 = df_stats_mes2
                st.session_state.df_gen_diario2 = df_gen_diario2

        if st.session_state.mostrar_graficas_wind:

            df_stats_mes2 = st.session_state.df_stats_mes2
            df_gen_diario2 = st.session_state.df_gen_diario2

            #Graficar
            grafica1(df_stats_mes2, df_gen_diario2, "Capturado Eolico y Generacion Unitaria Diario")

        if results["bess_factor"] == 1 and (results["nodo_bess"] != st.session_state.get("nodo_bess_anterior") or results["anios_analisis"] != st.session_state.get("anios_analisis_anterior") or results["anio_inicio"] != st.session_state.get("anio_inicio_anterior")):

                st.session_state.nodo_bess_anterior = results["nodo_bess"]
                st.session_state.mostrar_graficas_bess = True

                status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Ingresos BESS...</span>",
                    unsafe_allow_html=True
                )
                time.sleep(1)

                #Generacion BESS
                df_bess_mensual3, df_gen_diario3, df_gen_mensual3 = Energia_BESS(ruta_modelinput, results["anios_analisis"], results["anio_inicio"])
                # Coste marginal con las hidrologias
                df_prec = cmg(ruta_modelinput, results["nodo_bess"], results["anio_inicio"], results["anios_analisis"])

                progreso = st.progress(0)
                texto = st.empty()

                resultados_BESS = []
                stats_BESS = []
                for k, h_bess in enumerate(range(1, 6), start=1):

                    def actualizar(i, kk=k, n=results["n_hydro"]):
                        progreso.progress(i / n)
                        texto.markdown(
                            f"**Calculando hidrologia: {i} / {n} - {kk} de 5**"
                        )

                    
                    #Ingresos BESS
                    df_ingresos_BESS, df_stats_mes3 = Revenue_BESS(df_bess_mensual3, results["n_hydro"], df_prec, h_bess, callback=actualizar)

                    resultados_BESS.append({
                        "h_bess": h_bess,
                        "df_ingresos": df_ingresos_BESS
                    })

                    stats_BESS.append({
                        "h_bess": h_bess,
                        "df_stats": df_stats_mes3
                    })

                #print(resultados_BESS)
                #print(stats_BESS)
                st.session_state.resultados_BESS = resultados_BESS
                st.session_state.df_gen_mensual3 = df_gen_mensual3
                st.session_state.df_gen_diario3 = df_gen_diario3
                st.session_state.stats_BESS = stats_BESS

        if st.session_state.mostrar_graficas_bess:

            df_gen_diario3 = st.session_state.df_gen_diario3
            stats_BESS = st.session_state.stats_BESS

            dfs_por_h = {d["h_bess"]: d["df_stats"] for d in stats_BESS}

            #Graficar
            for k, h_bess in enumerate(range(1, 6), start=1):
                
                df_stats_mes3 = dfs_por_h[h_bess]
                #print(h_bess)
                #print(df_stats_mes3)
                grafica1(df_stats_mes3, df_gen_diario3, f"Capturado BESS y Generacion Unitaria Diaria - {h_bess}h")
    




        st.session_state.anio_inicio_anterior = results["anio_inicio"]
        st.session_state.anios_analisis_anterior= results["anios_analisis"]

        df_ingresos_gen = st.session_state.df_ingresos_gen
        df_gen_mensual = st.session_state.df_gen_mensual

        df_ingresos_gen2 = st.session_state.df_ingresos_gen2
        df_gen_mensual2 = st.session_state.df_gen_mensual2

        resultados_BESS = st.session_state.resultados_BESS
        df_gen_mensual3 = st.session_state.df_gen_mensual3



        return df_ingresos_gen, df_ingresos_gen2, resultados_BESS, df_gen_mensual, df_gen_mensual2, df_gen_mensual3
    
#----------------------------------------- GENERAR COMBINACICONES --------------------------------------------#

def generate_combinations(solar_factor, wind_factor, bess_factor, step, high):
    
            status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Formando combinaciones de PV+WIND+BESS...</span>",
                    unsafe_allow_html=True
            )
            time.sleep(1)

            # Generar los rangos de los parametros
            pv_values = np.arange(step, high + 1, step) if solar_factor == 1 else [0]  # Si solar_factor es 0, solo 0
            wind_values = np.arange(step, high + 1, step) if wind_factor == 1 else [0]  # Si wind_factor es 0, solo 0
            bess_values = np.arange(step, high + 1, step) if bess_factor == 1 else [0]  # Si bess_factor es 0, solo 0
            hours_values = np.arange(1, 6, 1) if bess_factor == 1 else [0]  # Si BESS es 0, solo 0 horas

            matriz1 = [(pv, 0, 0, 0) for pv in pv_values]
            matriz2 = [(0, wind, 0, 0) for wind in wind_values]
            matriz3 = [(pv, wind, 0, 0) for pv, wind in itertools.product(pv_values, wind_values)]
            matriz4 = [(pv, 0, bess, h) for pv, bess, h in itertools.product(pv_values, bess_values, hours_values)]
            matriz5 = [(0, 0, bess, h) for bess, h in itertools.product(bess_values, hours_values)]
    

            # Generar todas las combinaciones posibles
            if solar_factor == 1 and wind_factor == 1 and bess_factor == 0:  # PV + WIND
                matriz = matriz1 + matriz2 + matriz3
            elif solar_factor == 1 and wind_factor == 0 and bess_factor == 1:  # PV + BESS
                matriz = matriz1 + matriz4 + matriz5
            elif solar_factor == 1 and wind_factor == 0 and bess_factor == 0:  # Solo PV
                matriz = matriz1
            elif solar_factor == 0 and wind_factor == 0 and bess_factor == 1:  # Solo BESS
                matriz = matriz5
            elif solar_factor == 0 and wind_factor == 1 and bess_factor == 0:  # Solo WIND
                matriz = matriz2
            else:
                matriz = matriz1 + matriz2 + matriz3 + matriz4 + matriz5


            # Unir todas las combinaciones en una lista    
            matriz_filtrada = [fila for fila in matriz if any(valor != 0 for valor in fila)]
            # Convertir en DataFrame
            df_combinations = pd.DataFrame( matriz_filtrada, columns=["PV (MW)", "WIND (MW)", "BESS (MW)", "BESS (h)"])


            # Anadir columna 'Combination Type'
            def determine_combination_type(row):
                pv, wind, bess = row["PV (MW)"], row["WIND (MW)"], row["BESS (MW)"]
                if pv > 0 and wind == 0 and bess == 0:
                    return 1
                elif pv == 0 and wind > 0 and bess == 0:
                    return 2
                elif pv == 0 and wind == 0 and bess > 0:
                    return 3
                elif pv > 0 and wind > 0 and bess == 0:
                    return 4
                elif pv > 0 and wind == 0 and bess > 0:
                    return 5
                return 0

            df_combinations["Config"] = df_combinations.apply(determine_combination_type, axis=1)   

            cols = ["Config", "PV (MW)", "WIND (MW)", "BESS (MW)", "BESS (h)"]
            st.markdown("## Combinaciones para Simular")
            st.dataframe(
                df_combinations[cols],
                width="stretch",
                hide_index=True
            )

            return df_combinations


#----------------------------------------------- MARGEN PLANTA ------------------------------------------------#

def Margen_Planta(results, combinations, df_ingresos_gen_PV, df_ingresos_gen_WIND, resultados_BESS, df_gen_mensual_PV, df_gen_mensual_WIND, df_gen_mensual_BESS):

            PV_MW_values = combinations['PV (MW)'].to_numpy().T
            WIND_MW_values = combinations['WIND (MW)'].to_numpy().T
            BESS_MW_values = combinations['BESS (MW)'].to_numpy().T
            BESS_h_values = combinations['BESS (h)'].to_numpy().T
            Config_values = combinations['Config'].to_numpy().T
            combinaciones_n = PV_MW_values.shape[0] # Numero de configuraciones

            from Calculo_modelo import NPV_Planta, NPV_InyeccionBESS, NPV_CosteBESS

            def Histograma_Margen(tecno, arr, df_ingresos_gen, df_gen_mensual, capturado, coste):

                wacc = results["wacc"]/100
                #Generacion
                if tecno != "BESS":
                    ingresos = df_ingresos_gen.loc[:, 1:results["n_hydro"]]
                    df_inyeccion = NPV_Planta(ingresos, arr, wacc, capturado, results["anios_analisis"])

                else:
                    df_inyeccion = NPV_InyeccionBESS(resultados_BESS, arr, BESS_h_values, wacc, capturado, results["n_hydro"], results["anios_analisis"])


                #Coste
                gen = df_gen_mensual[["GENERACION"]]
                if tecno != "BESS":                    
                    df_Coste = NPV_Planta(gen, arr, wacc, coste, results["anios_analisis"])

                else:
                    df_Coste = NPV_CosteBESS(gen, arr, BESS_h_values, wacc, coste, results["anios_analisis"])

                df_Coste_exp = pd.DataFrame(np.tile(df_Coste.values, (results["n_hydro"], 1)))
                df_Margen = df_inyeccion - df_Coste_exp

                #Margen unitario
                if tecno == "BESS":
                    selec_hbess = 5
                    mask = BESS_h_values == selec_hbess
                    indices = np.where(mask)[0]
                    pos = indices[np.argmax(arr[mask])]
                    valor_max = arr[pos]
                else:
                    valor_max = arr.max()
                    pos = np.where(arr == valor_max)[0][0]

                df_col = df_Margen[[pos]].copy()
                margen_unitario = df_col[pos] / valor_max

                #Graficar
                df_hist = pd.DataFrame({
                    "margen": margen_unitario.values
                })

                title0 = (
                    f"Histograma Margen Unitario {tecno} 5h: "
                    f"Inyeccion - Costo de Generacion (LCOE); "
                    f"factor penalizacion a la inyeccion 24x12 = {capturado * 100}%"
                )

                if tecno != "BESS":
                    titulo_0 = f"Margen unitario {tecno} [$/MW]"
                else:
                    titulo_0 = f"Margen unitario {tecno} [$/MWh]"

                chart = alt.Chart(df_hist).mark_bar().encode(
                    x=alt.X(
                        "margen:Q",
                        bin=alt.Bin(maxbins=20),
                        title=titulo_0
                    ),
                    y=alt.Y(
                        "count():Q",
                        title="Frecuencia"
                    )
                ).properties(
                    title=title0
                )

                st.altair_chart(chart, width="stretch")

                return df_Margen

            if results["solar_factor"] == 1:

                status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Margen Planta PV...</span>",
                    unsafe_allow_html=True
                )

                capturado_PV = 100/100
                df_Margen_PV = Histograma_Margen("PV", PV_MW_values, df_ingresos_gen_PV, df_gen_mensual_PV, capturado_PV, results["coste_pv"])

            if results["wind_factor"] == 1:

                status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Margen Planta WIND...</span>",
                    unsafe_allow_html=True
                )

                capturado_WIND = 87.7/100
                df_Margen_WIND = Histograma_Margen("WIND", WIND_MW_values, df_ingresos_gen_WIND, df_gen_mensual_WIND, capturado_WIND, results["coste_wind"])


            if results["bess_factor"] == 1:

                status_text.markdown(
                    "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Margen Planta BESS...</span>",
                    unsafe_allow_html=True
                )

                capturado_BESS = 100/100
                df_Margen_BESS = Histograma_Margen("BESS", BESS_MW_values, resultados_BESS, df_gen_mensual_BESS, capturado_BESS, results["coste_bess"])
                
            if results["solar_factor"] == 0:
                df_Margen_PV = pd.DataFrame(np.zeros((results["n_hydro"], combinaciones_n)))

            if results["wind_factor"] == 0:
                df_Margen_WIND = pd.DataFrame(np.zeros((results["n_hydro"], combinaciones_n)))

            if results["bess_factor"] == 0:
                df_Margen_BESS = pd.DataFrame(np.zeros((results["n_hydro"], combinaciones_n)))

            df_Margen_Planta = df_Margen_PV + df_Margen_WIND + df_Margen_BESS

            return df_Margen_Planta


#------------------------------------------------ MARGEN PPA ---------------------------------------------------#

def Margen_PPA(combinations):

    PV_MW_values = combinations['PV (MW)'].to_numpy().T
    combinaciones_n = PV_MW_values.shape[0] # Numero de configuraciones

    from Calculo_modelo import PPA, cmg

    status_text.markdown(
            "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Margen Contrato PPA...</span>",
            unsafe_allow_html=True
    )

    # RETIRO & CONTRATO
    wacc = results["wacc"]/100
    df_precios = cmg(ruta_modelinput, results["nodo_retiro"], results["anio_inicio"], results["anios_analisis"])
    df_Retiro_PPA, df_Consumo_PPA, df_stats_mes, df_Contrato_PPA, ppa_mw = PPA(df_precios, df_ppa_long, results["n_hydro"], wacc, results["ppa_price"], results["anios_analisis"])

    results.update({"ppa_mw": ppa_mw})

    #df_Retiro_PPA_exp = pd.DataFrame(np.tile(df_Retiro_PPA.values, (1, combinaciones_n)))

    #Graficar
    grafica1(df_stats_mes, df_Consumo_PPA, "Capturado y Consumo diario - PPA")

    #Margen
    df_Margen_PPA = df_Contrato_PPA - df_Retiro_PPA
    df_Margen_PPA_exp = pd.DataFrame(np.tile(df_Margen_PPA.values, (1, combinaciones_n)))

    #Graficar
    df_hist = pd.DataFrame({
    "margen": df_Margen_PPA.values.ravel()
    })

    title0 = "Histograma Margen PPA: Contrato - Retiro"

    chart = alt.Chart(df_hist).mark_bar().encode(
        x=alt.X(
            "margen:Q",
            bin=alt.Bin(maxbins=20),
            title="Margen [$]"
        ),
        y=alt.Y(
            "count():Q",
            title="Frecuencia"
        )
    ).properties(
        title=title0
    )

    st.altair_chart(chart, width="stretch")

    return df_Margen_PPA_exp

def Margen_Proyecto(combinations, df_Margen):

    status_text.markdown(
            "Estado Simulacion: <span style='color: orange; font-weight: bold;'>Calculando Margen Proyecto...</span>",
            unsafe_allow_html=True
    )

    PV_MW_values = combinations['PV (MW)'].to_numpy().T
    WIND_MW_values = combinations['WIND (MW)'].to_numpy().T
    BESS_MW_values = combinations['BESS (MW)'].to_numpy().T
    BESS_h_values = combinations['BESS (h)'].to_numpy().T
    Config_values = combinations['Config'].to_numpy().T
    combinaciones_n = PV_MW_values.shape[0] # Numero de configuraciones

     # Calcular los percentiles
    percentil_5 = np.percentile(df_Margen.values, 5, axis=0)   # Percentil 5%
    percentil_50 = np.percentile(df_Margen.values, 50, axis=0) # Percentil 50% (mediana)
    percentil_95 = np.percentile(df_Margen.values, 95, axis=0) # Percentil 95%
    dispersion = percentil_5 - percentil_50

    # Matriz de los percentiles por CONFIG
    matriz_percentil = [PV_MW_values, WIND_MW_values, BESS_MW_values, BESS_h_values, (percentil_50), (dispersion) , (percentil_5), Config_values]
    df_matriz_final = pd.DataFrame(matriz_percentil).T
    df_matriz_final.columns = ["MW PV","MW WIND","MW BESS", "h BESS", "EBITDA MEDIO", "DISPERSION", "TAIL", "CONFIG"]
    print(df_matriz_final)
    pd.set_option('display.max_columns', None)  # Muestra todas las columnas

    return df_matriz_final
#-------------------------------- EJECUTAR -----------------------------------#

# --- Parte central de graficas ---
if ejecutar:
    
    st.session_state.results = results

    try:
        with tab0:

            df_ingresos_gen_pv, df_ingresos_gen_wind, resultados_BESS, df_gen_mensual, df_gen_mensual2, df_gen_mensual3 = Calculo_IngresoUnitario(results)
            
        with tab1:
            combinations = generate_combinations(results["solar_factor"], results["wind_factor"], results["bess_factor"], step=10, high=300)
       
        with tab2:
            df_Margen_Planta = Margen_Planta(results, combinations, df_ingresos_gen_pv, df_ingresos_gen_wind, resultados_BESS, df_gen_mensual, df_gen_mensual2, df_gen_mensual3)
            #df_Margen_Planta.to_excel(r"C:\Users\IonHualdeIriondo\source\repos\Model PortOptm\Model PortOptm\df_Margen_Planta.xlsx", index=False)
            print(df_Margen_Planta)

        with tab3:
            df_Margen_PPA = Margen_PPA(combinations)            
            print(df_Margen_PPA)

        with tab4:
            df_Margen = df_Margen_Planta + df_Margen_PPA
            print(df_Margen)
            df_matriz_final = Margen_Proyecto(combinations, df_Margen)
            llamada_graficar(results, df_matriz_final)

    except Exception as e:
        st.error(f"Ocurrio un error al ejecutar el script: {e}")


    # ------------------------ GRAFICAR ---------------------------#

    #llamada_graficar()
    

