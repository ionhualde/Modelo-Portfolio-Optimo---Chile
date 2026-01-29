import pandas as pd
import numpy as np
import calendar
import itertools
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import os
import json
import subprocess
import streamlit as st
import altair as alt
import streamlit as st



##################### COMBINACIONES ##########################

def generate_combinations(solar_factor, wind_factor, bess_factor, step, high):
    
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

    return df_combinations
    


##################### NPV ##########################

def NPV(df,tasa_descuento_anual):

    # Parametros   
    tasa_descuento_mensual = (1 + tasa_descuento_anual) ** (1/12) - 1

    # Supongamos que ya tienes el DataFrame `df_resultado` de dimensiones (180x120)
    valores = df.values  # Convertir DataFrame en array de NumPy

    # Crear un rango de periodos desde 0 hasta 179 (filas 0 a 179)
    periodos = np.arange(180).reshape(-1, 1)  # Redimensionar para broadcasting

    # Calcular el NPV por columna
    npv_por_columna = np.sum(valores / ((1 + tasa_descuento_mensual) ** periodos), axis=0)

    # Convertir el resultado en un DataFrame (1x120)
    df_npv = pd.DataFrame([npv_por_columna], columns=df.columns)
    return df_npv


##################### RETIRO ##########################

def Retiro(nodo_retiro, PPA_MW, df_fecha, tasa_descuento_anual, estructura_ppa, horas):

    # Leer el archivo Excel y usar la primera fila como encabezado
    df = pd.read_excel(file_path, sheet_name=nodo_retiro, header=0)
    
    # Encontrar las columnas de "1"
    s1_col = df.columns.get_loc(f"{estructura_ppa} - 1")
    cols_seleccionadas = df.columns[s1_col : s1_col + 120]

    # Crear DataFrame con esas columnas y primeras 180 filas
    df_CMgRetiro = df.loc[:179, cols_seleccionadas].copy()

    # Multiplicar cada valor de df1 con las columnas de df2
    df_Retiro = df_CMgRetiro.mul(df_fecha["Dias_mes"], axis=0)*PPA_MW*horas

    # NPV
    df_Retiro_NPV = NPV(df_Retiro,tasa_descuento_anual).T
    df_Retiro_NPV = df_Retiro_NPV.reset_index(drop=True)

    return df_Retiro_NPV

##################### INYECCION ##########################

def Inyeccion(df_inyeccion, MW_values,tasa_descuento_anual):

    # Calcular el NPV para cada valor de MW
    npv_matriz = []
    i = 0
    for MW in MW_values:
        df_mw = df_inyeccion * MW       

        # NPV
        df_NPV = NPV(df_mw,tasa_descuento_anual)
        npv_matriz.append(df_NPV.values.flatten())

    df_Inyeccion_NPV = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120
    
    return df_Inyeccion_NPV

##################### INYECCION BESS ##########################

def Inyeccion_BESS(nombre, MW_values, h_values, tasa_descuento_anual, corridas, capturado):

    # Leer el archivo Excel
    df = pd.read_excel(file_path, sheet_name="Inyeccion BESS", header=0)

    # Identificar las columnas necesarias
    value_col = df.columns.get_loc(f"{nombre} - 1")  # Columna con los datos
    t = 180 # 15 anos * 12 meses

    # Calcular el NPV para cada valor de MW
    npv_matriz = []
    for i, MW in enumerate(MW_values):

        bess_h = h_values[i]
        # Extraer las columnas REVENUES
        col = value_col + bess_h - 1
        df_value1 = df.iloc[:, col] * capturado
        reshaped_data1 = np.reshape(df_value1, (corridas, t)).T
        df_injeccion1 = pd.DataFrame(reshaped_data1)
        df_mw = df_injeccion1 * MW        

        # NPV
        df_NPV = NPV(df_mw,tasa_descuento_anual)
        npv_matriz.append(df_NPV.values.flatten())

    df_Inyeccion_NPV = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120

    return df_Inyeccion_NPV

##################### PPA ##########################

def PPA(PPA_MW,PPA_Price,df_num_days,tasa_descuento_anual, horas):

    
    df_PPA = df_num_days*PPA_MW*horas*PPA_Price
    df_PPA.columns = ['Ingreso PPA [$/mes]']

    # NPV
    df_PPA_NPV = pd.DataFrame(NPV(df_PPA,tasa_descuento_anual))
    
    return df_PPA_NPV

##################### COSTES ##########################


def Coste_Planta(nombre_planta,Coste_Planta,MW_values,perfiles,tasa_descuento_anual, df_num_days):


    # Leer el archivo Excel
    df = pd.read_excel(file_path, sheet_name=perfiles, header=0)

    # Identificar las columnas necesarias
    Gen_col = df.columns.get_loc(nombre_planta)
    # Extraer las columnas
    Gen_month = pd.DataFrame(df.iloc[:180, Gen_col])   

    df_Coste = Gen_month*Coste_Planta
 
    df_Coste.columns = ['Coste [$/mes]']

    # Calcular el NPV para cada valor de MW
    npv_matriz = []

    for MW in MW_values:

        # Multiplicar por PPA_MW
        df_mw = df_Coste * MW 
    
        # NPV
        df_NPV = NPV(df_mw,tasa_descuento_anual)
        npv_matriz.append(df_NPV.values.flatten())

    df_Coste_NPV = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120

    return df_Coste_NPV
    


##################### COSTE BESS ##########################

def Coste_bess(nombre_bess, coste_bess, MW_values, h_values, df_num_days, tasa_descuento_anual):

    # Leer el archivo Excel
    df = pd.read_excel(file_path, sheet_name="Perfiles BESS", header=0)

     # Identificar las columnas necesarias
    Gen_col = df.columns.get_loc(nombre_bess)

     # Extraer las columnas
    df_GWh = pd.DataFrame(df.iloc[:180, Gen_col])

    # Identificar las columnas necesarias
    #df_LCOEbess = pd.DataFrame(coste_bess)
    df_Coste = df_GWh.mul(df_num_days["Dias_mes"], axis=0)

    # Calcular el NPV para cada valor de MW
    npv_matriz = []
    for i, MW in enumerate(MW_values):

        bess_h = h_values[i]
        if bess_h == 0:
            cost = 0
        else:
            cost = coste_bess[bess_h-1]

        df_mw = df_Coste * MW * cost * bess_h

        # NPV
        df_NPV = NPV(df_mw,tasa_descuento_anual)
        npv_matriz.append(df_NPV.values.flatten())

    df_Coste_NPV = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120

    return df_Coste_NPV


################################################    MAIN   ########################################################

def MAIN():


    with open("C:/Users/ihualde/source/repos/Model PortOptm/Model PortOptm/inputs.json", "r") as f:
        results = json.load(f)

    # Extraer variables
    anio_inicio = results["anio_inicio"]
    ano_tot = results["anios_analisis"]

    nombre_pv = results["nombre_pv"]
    nodo_pv = results["nodo_pv"]
    coste_pv = results["coste_pv"]

    nombre_wind = results["nombre_wind"]
    nodo_wind = results["nodo_wind"]
    coste_wind = results["coste_wind"]

    nodo_bess = results["nodo_bess"]
    coste_bess2 = results["coste_bess"]
    coste_bess = [coste_bess2[f"{h}h"] for h in range(5, 0, -1)]

    ppa_mw = results["ppa_mw"]
    ppa_price = results["ppa_price"]
    nodo_retiro = results["nodo_retiro"]
    estructura_ppa = results["estructura_ppa"]

    solar_factor = results["solar_factor"]
    wind_factor = results["wind_factor"]
    bess_factor = results["bess_factor"]

    wacc = results["wacc"]

    #horas PPA
    if estructura_ppa == "ABC":
        horas = 24
    elif estructura_ppa == "AC":
        horas = 14
    else:
        horas = 10

    #-------------------------------- Combinaciones ------------------------------------#

    combinations = generate_combinations(solar_factor, wind_factor, bess_factor, step, high)
    PV_MW_values = combinations['PV (MW)'].to_numpy().T
    WIND_MW_values = combinations['WIND (MW)'].to_numpy().T
    BESS_MW_values = combinations['BESS (MW)'].to_numpy().T
    BESS_h_values = combinations['BESS (h)'].to_numpy().T
    Config_values = combinations['Config'].to_numpy().T
    combinaciones_n = PV_MW_values.shape[0] # Numero de configuraciones
    #print(combinations)

    #----------------------------- MODELO OPTIMO DE PORTFOLIO ----------------------------------#





##################### INPUTS #############################

corridas = 120
step = 10
high = 300

# Leer el archivo de Excel con los datos (ajusta la ruta al archivo de tu sistema)
file_path = 'C:\\Users\\ihualde\\source\\repos\\Model PortOptm\\Model PortOptm\\Model_Input.xlsx'

df_matriz_final = MAIN()


