######################### GRAFICAR #########################
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
import streamlit as st
import altair as alt

def GRAPH_EBITDA_DISP(df_matriz_final):

    # Crear un mapa de colores para las configuraciones con etiquetas descriptivas
    colors_dict = {
        1: 'blue',   # PV
        2: 'gray',  # WIND
        3: 'red',    # BESS
        4: 'green', # PV + WIND
        5: 'orange', # PV + BESS
    }
    labels_dict = {
        1: 'PV',
        2: 'WIND',
        3: 'BESS',
        4: 'PV + WIND',
        5: 'PV + BESS',
    }

    df_matriz_final = df_matriz_final[df_matriz_final["CONFIG"] != 0]  # Eliminar filas donde CONFIG = 0

    # Asignar colores
    df_matriz_final = df_matriz_final.iloc[1:].reset_index(drop=True)
    cmap = [colors_dict[config] for config in df_matriz_final["CONFIG"]]

    # Tamano mas pequeno para las configuraciones "0"
    sizes = df_matriz_final["CONFIG"].apply(lambda x: 30 if x == 0 else 50 + (x * 10))

    # Configuracion del grafico interactivo
    fig, ax = plt.subplots(figsize=(12, 8))

    # Crear el scatter plot
    scatter = ax.scatter(
        df_matriz_final["DISPERSION"],
        df_matriz_final["EBITDA MEDIO"],
        c=cmap,
        s=sizes,
        edgecolors='k',
        alpha=0.8
    )

    # Etiquetas de los ejes
    ax.set_xlabel("DISPERSION (M$)", fontsize=14)
    ax.set_ylabel("EBITDA MEDIO (M$)", fontsize=14)
    ax.set_title("Relacion entre DISPERSION y EBITDA MEDIO con CONFIGURACIONES", fontsize=16)

    # Anadir cuadricula
    ax.grid(visible=True, linestyle="--", alpha=0.6)

    # Anadir leyenda con etiquetas descriptivas
    legend_labels = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=10, label=label)
                     for label, color in zip(labels_dict.values(), colors_dict.values())]
    ax.legend(handles=legend_labels, title="Configuraciones", loc="upper right")

    # Anadir interactividad
    annot = ax.annotate("", xy=(0, 0), xytext=(10, 10),
                        textcoords="offset points", bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        index = ind["ind"][0]
        point = df_matriz_final.iloc[index]
        text = (f"PV: {point['MW PV']} MW\n"
                f"WIND: {point['MW WIND']} MW\n"
                f"BESS: {point['MW BESS']} MW\n"
                f"h BESS: {point['h BESS']} h\n"
                f"Config: {labels_dict[point['CONFIG']]}")
        annot.xy = (point["DISPERSION"], point["EBITDA MEDIO"])
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(colors_dict[point["CONFIG"]])
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = scatter.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            elif vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Mostrar el grafico
    plt.tight_layout()
    st.pyplot(fig)
    #plt.show()

def HEATMAP_PV_WIND(df_matriz_final, step, PPA_MW, high):    

    df = df_matriz_final        
    # Crear una fila de ceros con el mismo numero de columnas
    fila_ceros = pd.DataFrame([0] * len(df.columns)).T
    fila_ceros.columns = df.columns

    # Anadir la fila de ceros al final del DataFrame
    df = pd.concat([df, fila_ceros], ignore_index=True)

    # Crear la matriz de EBITDA en funcion de MW PV y MW WIND
    pv_values  = wind_values = np.arange(step, high + 1, step)

    # Inicializamos la matriz de EBITDA
    ebitda_matrix = np.zeros((len(pv_values), len(wind_values)))
    disp_matrix = np.zeros((len(pv_values), len(wind_values)))

    # Rellenar la matriz con los valores de EBITDA
    for i, pv in enumerate(pv_values):
        for j, wind in enumerate(wind_values):
            # Filtramos el dataframe para obtener el EBITDA correspondiente a la combinacion de PV y WIND
            ebitda_value = df[(df["MW PV"] == pv) & (df["MW WIND"] == wind)]["EBITDA MEDIO"].values[0]
            ebitda_matrix[i, j] = ebitda_value
            disp_value = df[(df["MW PV"] == pv) & (df["MW WIND"] == wind)]["DISPERSION"].values[0]
            disp_matrix[i, j] = disp_value

    # Crear el grafico
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # Tres subgraficas en una fila

    # Seleccionar cada segundo valor para mostrar
    x_axis = np.round(pv_values / PPA_MW,1)
    y_axis = np.round(wind_values / PPA_MW,1)

    div = 4
    # Seleccionar cada segundo valor para mostrar
    x_axis_labels = x_axis[::div]
    y_axis_labels = y_axis[::div]

    # Graficar EBITDA
    ax = axes[0]
    im = ax.imshow(ebitda_matrix, cmap="RdYlGn")
    ax.set_title("EBITDA MEDIO")
    # Configurar etiquetas de los ejes con saltos
    ax.set_xticks(np.arange(0, len(x_axis)+1, div))  # Posiciones de los ticks (cada dos)
    ax.set_yticks(np.arange(0, len(y_axis)+1, div))  # Posiciones de los ticks (cada dos)
    # Asignar las etiquetas correspondientes
    ax.set_xticklabels(x_axis_labels)
    ax.set_yticklabels(y_axis_labels)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=10)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=10)
    fig.colorbar(im, ax=ax, orientation="horizontal", label="EBITDA (M$)")
    ax.set_xlabel("MW WIND / MW PPA")
    ax.set_ylabel("MW PV / MW PPA")

    # Graficar DISP
    ax = axes[1]
    im = ax.imshow(disp_matrix, cmap="RdYlGn")
    ax.set_title("DISPERSION")
    # Configurar etiquetas de los ejes con saltos
    ax.set_xticks(np.arange(0, len(x_axis)+1, div))  # Posiciones de los ticks (cada dos)
    ax.set_yticks(np.arange(0, len(y_axis)+1, div))  # Posiciones de los ticks (cada dos)
    # Asignar las etiquetas correspondientes
    ax.set_xticklabels(x_axis_labels)
    ax.set_yticklabels(y_axis_labels)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True, pad=10)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True, pad=10)
    fig.colorbar(im, ax=ax, orientation="horizontal", label="DISPERSION (M$)")
    ax.set_xlabel("MW WIND / MW PPA")
    ax.set_ylabel("MW PV / MW PPA")

    plt.tight_layout()
    st.pyplot(fig)
    #plt.show()

def HEATMAP_PV_BESS(df_matriz_final, step, PPA_MW, high):    

    # Valores de h BESS para iterar
    h_bess_values = [5, 4, 3, 2, 1]

    for h_bess in h_bess_values:

        df = df_matriz_final[df_matriz_final["h BESS"] == h_bess].copy()

        fila_ceros = pd.DataFrame([0] * len(df.columns)).T
        fila_ceros.columns = df.columns
        df = pd.concat([df, fila_ceros], ignore_index=True)

        pv_values = np.arange(step, high + 1, step)
        bess_values = np.arange(step, high + 1, step)

        ebitda_matrix = np.zeros((len(pv_values), len(bess_values)))
        disp_matrix = np.zeros((len(pv_values), len(bess_values)))

        for i, pv in enumerate(pv_values):
            for j, bess in enumerate(bess_values):
                mask = (df["MW PV"] == pv) & (df["MW BESS"] == bess)
                ebitda_matrix[i, j] = df.loc[mask, "EBITDA MEDIO"].values[0]
                disp_matrix[i, j] = df.loc[mask, "DISPERSION"].values[0]

        df_ebitda = pd.DataFrame({
            "MW_PV": np.repeat(pv_values, len(bess_values)),
            "MW_BESS": np.tile(bess_values, len(pv_values)),
            "VALOR": ebitda_matrix.flatten()
        })

        df_disp = pd.DataFrame({
            "MW_PV": np.repeat(pv_values, len(bess_values)),
            "MW_BESS": np.tile(bess_values, len(pv_values)),
            "VALOR": disp_matrix.flatten()
        })

        for d in [df_ebitda, df_disp]:
            d["MW_PV_PPA"] = (d["MW_PV"] / PPA_MW).round(1)
            d["MW_BESS_PPA"] = (d["MW_BESS"] / PPA_MW).round(1)

        # Rangos independientes
        ebitda_min, ebitda_max = df_ebitda["VALOR"].min(), df_ebitda["VALOR"].max()
        disp_min, disp_max = df_disp["VALOR"].min(), df_disp["VALOR"].max()

        heat_ebitda = alt.Chart(df_ebitda).mark_rect().encode(
            x=alt.X("MW_BESS_PPA:O", title="MW BESS / MW PPA"),
            y=alt.Y("MW_PV_PPA:O", title="MW PV / MW PPA"),
            color=alt.Color(
                "VALOR:Q",
                scale=alt.Scale(
                    domain=[ebitda_min, ebitda_max],
                    scheme="redyellowgreen"
                ),
                title="EBITDA (M$)"
            ),
            tooltip=["MW_PV_PPA", "MW_BESS_PPA", "VALOR"]
        ).properties(
            title=f"EBITDA - {h_bess}h",
            width=350,
            height=350
        )

        heat_disp = alt.Chart(df_disp).mark_rect().encode(
            x=alt.X("MW_BESS_PPA:O", title="MW BESS / MW PPA"),
            y=alt.Y("MW_PV_PPA:O", title="MW PV / MW PPA"),
            color=alt.Color(
                "VALOR:Q",
                scale=alt.Scale(
                    domain=[disp_min, disp_max],
                    scheme="redyellowgreen"
                ),
                title="DISPERSION (M$)"
            ),
            tooltip=["MW_PV_PPA", "MW_BESS_PPA", "VALOR"]
        ).properties(
            title=f"DISPERSION - {h_bess}h",
            width=350,
            height=350
        )

        st.altair_chart(
            (heat_ebitda | heat_disp).resolve_scale(color="independent"),
            width="stretch"
        )

def GRAPH_BESS(df_matriz_final, step, PPA_MW):  
    
    ####################################### h BESS & MW BESS ###############################################

    df_1 = df_matriz_final

    # Supongamos que df_1 es el DataFrame inicial
    i2_values = [50, 100, 200, 300]  # Los valores de MW PV a usar

    # Filtrado paso a paso para evitar reindexado incorrecto
    df_filtered = df_1[(df_1['MW BESS'] != 0) & (df_1['h BESS'] != 0) & (df_1['MW WIND'] == 0)]

    # Crear la figura y los subgraficos (4 filas x 2 columnas)
    fig, axes = plt.subplots(4, 2, figsize=(12, 18))  # Aseguramos que haya 8 subgraficos

    # Iterar sobre los diferentes valores de i2 (50, 100, 200, 300)
    for idx, i2 in enumerate(i2_values):

        i2_condition = df_1['MW PV'] == i2
        # Reindexar la condicion booleana para que tenga el mismo indice que df_filtered
        i2_condition = i2_condition.reindex(df_filtered.index, fill_value=False)

        # Aplicar ambas condiciones
        data_filter = df_filtered[i2_condition]

        # Crear el pivot para EBITDA MEDIO
        pivot_df_ebitda = data_filter.pivot(index='h BESS', columns='MW BESS', values='EBITDA MEDIO')
        X = pivot_df_ebitda.columns
        Y = pivot_df_ebitda.index
        Z_ebitda = pivot_df_ebitda.values
        Z_ebitda = Z_ebitda / 1000

        # Crear el pivot para DISPERSION
        pivot_df_disp = data_filter.pivot(index='h BESS', columns='MW BESS', values='DISPERSION')
        Z_disp = pivot_df_disp.values
        Z_disp = Z_disp / 1000

        # Obtener el indice de fila para el subplot (filas 0-3)
        row = idx  # Para asignar las filas 0, 1, 2, 3
        # Columna 0 para EBITDA y columna 1 para DISPERSION
        col_ebitda = 0
        col_disp = 1

        # Grafico de contorno de EBITDA MEDIO
        CS_ebitda = axes[row, col_ebitda].contourf(X, Y, Z_ebitda)
        axes[row, col_ebitda].clabel(CS_ebitda, inline=True, fontsize=10)
        axes[row, col_ebitda].set_title(f'EBITDA MEDIO - MW PV = {i2}')
        axes[row, col_ebitda].set_xlabel('MW BESS')
        axes[row, col_ebitda].set_ylabel('h BESS')
        axes[row, col_ebitda].set_xlim(10, 300)  # Limitar el eje X de 10 a 300
        axes[row, col_ebitda].set_ylim(1, 5)     # Limitar el eje Y de 1 a 5
        fig.colorbar(CS_ebitda, ax=axes[row, col_ebitda], label="EBITDA (M$)")  # Dividir entre 1000

        # Grafico de contorno de DISPERSION
        CS_disp = axes[row, col_disp].contourf(X, Y, Z_disp)
        axes[row, col_disp].clabel(CS_disp, inline=True, fontsize=10)
        axes[row, col_disp].set_title(f'DISPERSION - MW PV = {i2}')
        axes[row, col_disp].set_xlabel('MW BESS')
        axes[row, col_disp].set_ylabel('h BESS')
        axes[row, col_disp].set_xlim(10, 300)  # Limitar el eje X de 10 a 300
        axes[row, col_disp].set_ylim(1, 5)     # Limitar el eje Y de 1 a 5
        fig.colorbar(CS_disp, ax=axes[row, col_disp], label="DISPERSION (M$)")  # Dividir entre 1000

    # Ajustar el espacio entre subgraficos
    plt.tight_layout()
    st.pyplot(fig)
    #plt.show()


