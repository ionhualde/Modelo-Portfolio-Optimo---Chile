import pandas as pd
import numpy as np
import calendar
import time

from pandas.io import excel    

def leer_ppa(ruta_ppa):

    #-------------------------------------------------#
    #Leer PPA    

    df_ppa = pd.read_excel(
        ruta_ppa,
        header=None,
        usecols="C:N",
        skiprows=1,
        nrows=24
    )

    # asignar indices claros
    df_ppa.index = range(0, 24)     # HORA 1-24
    df_ppa.columns = range(1, 13)   # MES 1-12

    # pasar a formato largo
    df_ppa_long = (
        df_ppa
        .reset_index()
        .melt(id_vars="index", var_name="MES", value_name="PPA_MW")
        .rename(columns={"index": "HORA"})
    )

    # por si acaso
    df_ppa_long["HORA"] = df_ppa_long["HORA"].astype(int)
    df_ppa_long["MES"] = df_ppa_long["MES"].astype(int)

    return df_ppa_long


def tabla_rer(ruta_modelinput):

    df = pd.read_excel(ruta_modelinput, sheet_name="RER")

    # Limpiar strings vacios
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # =====================
    # TABLA 1 (6 columnas)
    # =====================
    df_tabla1 = (
        df.loc[df["Projecto"].notna(), df.columns[:6]]
          .reset_index(drop=True)
    )

    # =====================
    # TABLA 2 (empieza en MES)
    # =====================

    # Fila donde empieza MES
    fila_inicio_tabla2 = df["MES"].first_valid_index()

    if fila_inicio_tabla2 is None:
        raise ValueError("No se encontro la columna MES con datos")

    # Columna donde empieza MES
    col_inicio_tabla2 = df.columns.get_loc("MES")

    # Cortar bloque completo
    df_tabla2 = df.iloc[fila_inicio_tabla2:, col_inicio_tabla2:]

    # Cortar hasta el primer NaN en MES
    idx_nan_mes = df_tabla2[df_tabla2["MES"].isna()].index
    if not idx_nan_mes.empty:
        df_tabla2 = df_tabla2.loc[:idx_nan_mes[0]-1]

    df_tabla2 = df_tabla2.reset_index(drop=True)

    return df_tabla1, df_tabla2


def generacion(planta, ano_tot, anio_inicio, df_tabla1, df_tabla2):

    factor = 0.985

    # -------- Buscar columna en tabla 2 (case-insensitive) --------
    col_planta = next(
        col for col in df_tabla2.columns
        if col.lower() == planta.lower()
    )

    # Subtabla base
    df_planta2 = df_tabla2[["MES", "DIA", "HORA", col_planta]].copy()
    df_planta2 = df_planta2.rename(columns={col_planta: "GENERACION"})

    # -------- Obtener capacidad desde tabla 1 --------
    capacidad = df_tabla1.loc[
        df_tabla1["Projecto"].str.lower() == planta.lower(),
        "Capacity MWac"
    ].iloc[0]

    #--------------------------------------------------#

    degradacion = 0.004  # 0.4%

    dfs = []

    for i in range(ano_tot):
        anio = anio_inicio + i
        factor = (1 - degradacion) ** i

        df_anio = df_planta2.copy()
        df_anio["ANO"] = anio
        df_anio["GENERACION"] = df_anio["GENERACION"] * factor

        dfs.append(df_anio)

    df_planta = pd.concat(dfs, ignore_index=True)

    # -------- Aplicar logica --------
    # 1) multiplicar por 98,5%
    n_alta = 0.985
    df_planta["GENERACION"] = df_planta["GENERACION"] * n_alta

    # 2) capear a capacidad
    df_planta["GENERACION"] = np.minimum(df_planta["GENERACION"], capacidad)

    # 3) normalizar (porcentual)
    df_planta["GENERACION"] = df_planta["GENERACION"] / capacidad

    # Agrupar por MES y HORA y promediar
    df_gen_tot = (
        df_planta.groupby(["ANO", "MES", "HORA"], as_index=False)["GENERACION"]
               .sum()
    )

    #--------------------------------------------------#


    #mensual
    df_gen_mensual = (
    df_gen_tot
    .groupby(["ANO", "MES"], as_index=False)["GENERACION"]
    .sum()
    )
    
    # df = tu DataFrame
    df_gen_diario = df_gen_mensual.copy()

    # Calcular dias del mes
    df_gen_diario["DIAS_MES"] = df_gen_diario.apply(
        lambda x: 28 if int(x["MES"]) == 2 else calendar.monthrange(int(x["ANO"]), int(x["MES"]))[1],
        axis=1
    )

    # Generacion diaria promedio
    df_gen_diario["GENERACION_DIARIA"] = df_gen_diario["GENERACION"] / df_gen_diario["DIAS_MES"]

    df_gen_diario = df_gen_diario.drop(columns=["GENERACION"])
    df_gen_diario = df_gen_diario.drop(columns=["DIAS_MES"])
    df_gen_diario = df_gen_diario.rename(columns={"GENERACION_DIARIA": "GENERACION"})


    return df_gen_tot, df_gen_mensual, df_gen_diario


def cmg(ruta_modelinput, barra, anio_inicio, ano_tot):

    # Leer hoja completa
    df_cmg = pd.read_excel(ruta_modelinput, sheet_name=barra)

    # Limpiar strings vacios
    df_cmg = df_cmg.replace(r'^\s*$', np.nan, regex=True)

    # Eliminar filas y columnas completamente vacias
    df_cmg = df_cmg.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Filtrar desde el ano de inicio
    df_prec = df_cmg[df_cmg["ANO"] >= anio_inicio].copy()

    # Calcular ano final que necesitamos
    anio_final = anio_inicio + ano_tot - 1

    # Anos disponibles en los datos
    anio_max = df_prec["ANO"].max()

    # Si necesitamos mas anos que los disponibles, repetimos el ultimo ano
    if anio_final > anio_max:
        anos_faltantes = anio_final - anio_max
        df_ultimo = df_prec[df_prec["ANO"] == anio_max].copy()
    
        # Repetir los ultimos precios tantas veces como anos faltan
        df_repetido = pd.concat([df_ultimo.assign(ANO=anio_max + i + 1) for i in range(anos_faltantes)], ignore_index=True)
    
        # Concatenar con los datos originales
        df_prec = pd.concat([df_prec, df_repetido], ignore_index=True)

    # Ahora df_prec tiene exactamente 20 anos empezando en 2027
    #pd.set_option('display.max_rows', None)
    #print(df_prec)

    #--------------------------------------------------#

    return df_prec


def Revenues_Generador(df_prec, df_gen_tot, df_gen_mensual):

    # Separar columnas de precios
    cols_precios = df_prec.columns.difference(["ANO", "MES", "HORA"])

    # Merge de generacion con precios por ANO, MES, HORA
    df_merge = pd.merge(
        df_gen_tot,
        df_prec,
        on=["ANO", "MES", "HORA"],
        how="left"
    )

    # Multiplicar GENERACION por todos los precios (vectorizado)
    df_merge[cols_precios] = df_merge[cols_precios].multiply(df_merge["GENERACION"], axis=0)

    # Sumar por MES
    df_ingresos= df_merge.groupby(["ANO", "MES"])[cols_precios].sum().reset_index()

    #-------------------------- CAPTURADO -----------------------------#

    # copiar df_ingresos_gen para no modificar el original
    df_capturado = df_ingresos.copy()

    # columnas de ingresos (todas menos ANO y MES)
    cols_ingresos = df_capturado.columns.difference(["ANO", "MES"])

    # dividir fila a fila por la generacion mensual
    df_capturado[cols_ingresos] = (
        df_capturado[cols_ingresos]
        .div(df_gen_mensual["GENERACION"].values, axis=0)
    )

    # columnas de escenarios 1..120
    cols_esc = df_capturado.columns.difference(["ANO", "MES"])

    df_stats_mes = df_capturado[["ANO", "MES"]].copy()

    df_stats_mes["PROM"] = df_capturado[cols_esc].mean(axis=1)
    df_stats_mes["P5"]   = df_capturado[cols_esc].quantile(0.05, axis=1)
    df_stats_mes["P95"]  = df_capturado[cols_esc].quantile(0.95, axis=1)

    return df_ingresos, df_stats_mes


def Energia_BESS(ruta_modelinput, ano_tot, anio_inicio):

    df_bess = pd.read_excel(ruta_modelinput, sheet_name="BESS")

    # Limpiar SOH (100,00% -> 1.0)
    df_bess["SOH"] = (
        df_bess["SOH"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    rows = []
    mes_global = 1

    for i in range(len(df_bess) - 1):
        soh_ini = df_bess.loc[i, "SOH"]
        soh_fin = df_bess.loc[i + 1, "SOH"]

        degrad_mensual = (soh_ini - soh_fin) / 12

        for mes in range(1, 13):
            soh_mes = soh_ini - degrad_mensual * (mes - 1)

            rows.append({
                "MES_GLOBAL": mes_global,
                "ANO": i + 1,
                "MES": mes,
                "SOH": soh_mes
            })

            mes_global += 1

    # DataFrame final (300 meses)
    df_soh_mensual = pd.DataFrame(rows)

    # cortar a anos_tot
    n_meses = ano_tot * 12
    df_soh_mensual = df_soh_mensual.iloc[:n_meses].reset_index(drop=True)

    # Sobreintalado
    factor_sobreinstalado = 1 + 8/100  # 1.08
    df_soh_mensual["SOH"] = df_soh_mensual["SOH"] * factor_sobreinstalado

    df_bess["RTE"] = (
        df_bess["RTE"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    rows = []
    mes_global = 1

    for i in range(len(df_bess)):
        rte_y = df_bess.loc[i, "RTE"]
        eta_y = np.sqrt(rte_y)

        for mes in range(1, 13):
            rows.append({
                "MES_GLOBAL": mes_global,
                "ANO": i + 1,
                "MES": mes,
                "RTE": rte_y,
                "ETA": eta_y
            })
            mes_global += 1

    df_rte_mensual = pd.DataFrame(rows)

    df_bess_mensual = df_soh_mensual.merge(
        df_rte_mensual[["MES_GLOBAL", "ETA"]],
        on="MES_GLOBAL",
        how="left"
    )

    df_bess_mensual["ENERGIA_INYECTADA"] = (
        df_bess_mensual["SOH"] * df_bess_mensual["ETA"]
    )

    df_bess_mensual["ENERGIA_CARGADA"] = (
        df_bess_mensual["SOH"] / df_bess_mensual["ETA"]
    )

    df_bess_mensual["ANO_CAL"] = anio_inicio + (df_bess_mensual["MES_GLOBAL"] - 1) // 12
    df_bess_mensual["MES_CAL"] = (df_bess_mensual["MES_GLOBAL"] - 1) % 12 + 1

    df_gen_diario = (
        df_bess_mensual[["ANO_CAL", "MES_CAL", "ENERGIA_INYECTADA"]]
        .rename(columns={
            "ANO_CAL": "ANO",
            "MES_CAL": "MES",
            "ENERGIA_INYECTADA": "GENERACION"
        })
    )

    # Calcular dias del mes
    df_gen_diario["DIAS_MES"] = df_gen_diario.apply(
        lambda x: 28 if int(x["MES"]) == 2 else calendar.monthrange(int(x["ANO"]), int(x["MES"]))[1],
        axis=1
    )

    # Generacion diaria promedio
    df_gen_mes = df_gen_diario.copy()
    df_gen_mes["GENERACION_MES"] = df_gen_diario["GENERACION"] * df_gen_diario["DIAS_MES"]

    df_gen_mes = df_gen_mes.drop(columns=["GENERACION"])
    df_gen_mes = df_gen_mes.drop(columns=["DIAS_MES"])
    df_gen_mes = df_gen_mes.rename(columns={"GENERACION_MES": "GENERACION"})

    return df_bess_mensual, df_gen_diario, df_gen_mes


def Revenue_BESS(df_bess_mensual, hydros, df_prec, h_bess, callback=None):

    df_bess_mes = df_bess_mensual[[
        "ANO_CAL", "MES_CAL",
        "ENERGIA_INYECTADA", "ENERGIA_CARGADA"
    ]].copy()

    df_bess_mes["E_INY_MES"] = df_bess_mes["ENERGIA_INYECTADA"] * h_bess
    df_bess_mes["E_CAR_MES"] = df_bess_mes["ENERGIA_CARGADA"] * h_bess

    def despacho_bess_mes_carga_descarga(df_prec, anio, mes, hid, e_iny_mes, e_car_mes):
        # Filtrar precios del mes
        df_mes = df_prec[
            (df_prec["ANO"] == anio) &
            (df_prec["MES"] == mes)
        ][["HORA", hid]].copy()

        df_mes.rename(columns={hid: "PRECIO"}, inplace=True)

        # ---------- DESCARGA (precios altos) ----------
        df_desc = df_mes.sort_values("PRECIO", ascending=False).reset_index(drop=True)

        horas_full = int(np.floor(e_iny_mes))
        resto = e_iny_mes - horas_full

        df_desc["P_INY_MW"] = 0.0

        if horas_full > 0:
            df_desc.loc[:horas_full - 1, "P_INY_MW"] = 1.0

        if resto > 0 and horas_full < len(df_desc):
            df_desc.loc[horas_full, "P_INY_MW"] = resto

        df_desc["INGRESO"] = df_desc["P_INY_MW"] * df_desc["PRECIO"]

        # ---------- CARGA (precios bajos) ----------
        df_car = df_mes.sort_values("PRECIO", ascending=True).reset_index(drop=True)

        horas_full = int(np.floor(e_car_mes))
        resto = e_car_mes - horas_full

        df_car["P_CAR_MW"] = 0.0

        if horas_full > 0:
            df_car.loc[:horas_full - 1, "P_CAR_MW"] = 1.0

        if resto > 0 and horas_full < len(df_car):
            df_car.loc[horas_full, "P_CAR_MW"] = resto

        df_car["COSTE"] = df_car["P_CAR_MW"] * df_car["PRECIO"]

        return df_desc, df_car

    resultados = []

    
    for hid in range(1, hydros+1):

        if callback is not None:
            callback(hid)

        for _, row in df_bess_mes.iterrows():

            anio = row["ANO_CAL"]
            mes = row["MES_CAL"]

            e_iny = row["E_INY_MES"]
            e_car = row["E_CAR_MES"]

            df_desc, df_car = despacho_bess_mes_carga_descarga(
                df_prec, anio, mes, hid, e_iny, e_car
            )

            energia_iny = df_desc["P_INY_MW"].sum()
            energia_car = df_car["P_CAR_MW"].sum()

            precio_iny = (
                df_desc["INGRESO"].sum() / energia_iny
                if energia_iny > 0 else 0
            )

            precio_car = (
                df_car["COSTE"].sum() / energia_car
                if energia_car > 0 else 0
            )

            resultados.append({
                "ANO": anio,
                "MES": mes,
                "HIDROLOGIA": hid,
                "PRECIO_INYECTADO": precio_iny,
                "PRECIO_CARGA": precio_car
            })

    df_resultados_hid = pd.DataFrame(resultados)
    #df_resultados_hid.to_excel(r"C:\Users\IonHualdeIriondo\source\repos\Model PortOptm\Model PortOptm\df_resultados_hid.xlsx", index=False)
    df_resultados_hidros = (
        df_resultados_hid
        .sort_values(
            by=["HIDROLOGIA", "ANO", "MES"],
            ascending=[True, True, True]
        )
        .reset_index(drop=True)
    )

    df_resultados = df_resultados_hidros.merge(
        df_bess_mes[[
            "ANO_CAL", "MES_CAL",
            "E_INY_MES", "E_CAR_MES"
        ]],
        left_on=["ANO", "MES"],
        right_on=["ANO_CAL", "MES_CAL"],
        how="left"
    )

    df_resultados["INGRESO_NETO_MES"] = (
        df_resultados["PRECIO_INYECTADO"] * df_resultados["E_INY_MES"]
        - df_resultados["PRECIO_CARGA"] * df_resultados["E_CAR_MES"]
    )

    #df_resultados.to_excel(r"C:\Users\IonHualdeIriondo\source\repos\Model PortOptm\Model PortOptm\rrr333.xlsx", index=False)

    df_resultados = df_resultados.drop(columns=["ANO_CAL", "MES_CAL"])

    #dias
    df_resultados["DIAS_MES"] = df_resultados.apply(
        lambda x: 28 if int(x["MES"]) == 2 else calendar.monthrange(int(x["ANO"]), int(x["MES"]))[1],
        axis=1
    )

    df_resultados["INGRESO_NETO_MES"] = (
        df_resultados["INGRESO_NETO_MES"] * df_resultados["DIAS_MES"]
    )

    df_ingreso_mensual2 = (
        df_resultados
        .groupby(["ANO", "MES", "HIDROLOGIA"], as_index=False)
        ["INGRESO_NETO_MES"]
        .sum()
    )
    #df_ingreso_mensual2.to_excel(r"C:\Users\IonHualdeIriondo\source\repos\Model PortOptm\Model PortOptm\df_ingreso_mensual2.xlsx", index=False)
    df_ingreso_mensual = df_ingreso_mensual2.sort_values("HIDROLOGIA", kind="stable").reset_index(drop=True)

    df_pivot = (
    df_ingreso_mensual
    .pivot(
        index=["ANO", "MES"],
        columns="HIDROLOGIA",
        values="INGRESO_NETO_MES"
    )
    .reset_index()
)
    df_pivot.columns.name = None
    df_pivot["ANO"] = df_pivot["ANO"].astype(int)
    df_pivot["MES"] = df_pivot["MES"].astype(int)
    #df_pivot.to_excel(r"C:\Users\IonHualdeIriondo\source\repos\Model PortOptm\Model PortOptm\df_pivot.xlsx", index=False)
   #-------------------------- CAPTURADO -----------------------------#

    # copiar df_ingresos_gen para no modificar el original
    df_capturado = df_pivot.copy()

    # columnas de ingresos (todas menos ANO y MES)
    cols_ingresos = df_capturado.columns.difference(["ANO", "MES"])

    # dividir fila a fila por la generacion mensual
    df_capturado[cols_ingresos] = (
        df_capturado[cols_ingresos]
        .div(df_bess_mensual["ENERGIA_INYECTADA"].values, axis=0)
    )

    # dias del mes por fila
    dias_mes = pd.to_datetime(
        df_capturado["ANO"].astype(str) + "-" +
        df_capturado["MES"].astype(str).str.zfill(2) + "-01"
    ).dt.days_in_month

    df_capturado[cols_ingresos] = (
        df_capturado[cols_ingresos]
        .div(dias_mes.values, axis=0)  # dividir por dias del mes
        .div(h_bess)                        # dividir entre h_bess
        )

    # columnas de escenarios 1..120
    cols_esc = df_capturado.columns.difference(["ANO", "MES"])

    df_stats_mes = df_capturado[["ANO", "MES"]].copy()

    df_stats_mes["PROM"] = df_capturado[cols_esc].mean(axis=1)
    df_stats_mes["P5"]   = df_capturado[cols_esc].quantile(0.05, axis=1)
    df_stats_mes["P95"]  = df_capturado[cols_esc].quantile(0.95, axis=1)

    return df_pivot, df_stats_mes


def PPA(df_precio, df_ppa_long, hydro, tasa_descuento_anual, ppa_price, anios_tot):

    # columnas de hidrologias (1 a 120)
    cols_hidro = list(range(1, hydro+1))

    #-------------------------------------------------#
    #retiro

    df_precio = df_precio.merge(
        df_ppa_long,
        on=["MES", "HORA"],
        how="left"
    )

    df_retiro = df_precio.copy()
    df_retiro[cols_hidro] = (df_retiro[cols_hidro].multiply(df_retiro["PPA_MW"], axis=0))
    df_retiro = df_retiro.drop(columns=["PPA_MW"])

    df_coste_retiro = (
        df_retiro
        .groupby(["ANO", "MES"], sort=False)[cols_hidro]
        .sum()
        .reset_index()
        .copy()
    )

    #-------------------------------------------------#
    #Capturados

    #Suma mensual
    df_ppa_1ano = (
        df_ppa_long
        .groupby("MES", as_index=False)["PPA_MW"]
        .sum()
    )

    df = df_coste_retiro.merge(
        df_ppa_1ano,
        on="MES",
        how="left"
    )

    df_capturado = df.copy()
    df_capturado[cols_hidro] = (df_capturado[cols_hidro].div(df_capturado["PPA_MW"], axis=0))
    df_capturado.drop(columns=["PPA_MW"], inplace=True)

    #PROM, P95-P5
    df_promedio = df_capturado.copy()
    # columnas de escenarios 1..120
    cols_esc = df_promedio.columns.difference(["ANO", "MES"])

    df_stats_mes = df_promedio[["ANO", "MES"]].copy()

    df_stats_mes["PROM"] = df_promedio[cols_esc].mean(axis=1)
    df_stats_mes["P5"]   = df_promedio[cols_esc].quantile(0.05, axis=1)
    df_stats_mes["P95"]  = df_promedio[cols_esc].quantile(0.95, axis=1)

    #-------------------------------------------------#
    #Multiplicar dias del mes

    #Quitar ANO_MES
    df_capturado = df_promedio.loc[:, 1:hydro]

    #dias mes
    dias_mes = pd.to_datetime(
        df_promedio["ANO"].astype(str) + "-" +
        df_promedio["MES"].astype(str).str.zfill(2) + "-01"
    ).dt.days_in_month
    dias_mes = dias_mes.where(df_promedio["MES"] != 2, 28)

    df_coste_retiro = df_coste_retiro.copy()
    df_coste_retiro[cols_hidro] = df_coste_retiro[cols_hidro].mul(dias_mes, axis=0)
    df_coste_retiro0 = df_coste_retiro.loc[:, 1:hydro]

    # NPV
    df_Retiro_NPV = NPV(df_coste_retiro0,tasa_descuento_anual, anios_tot).T
    df_Retiro_NPV = df_Retiro_NPV.reset_index(drop=True)

    #------------------- CONTRATO -------------------#

    #Calcular MW promedio del anio
    df_consumo_ppa = df_ppa_1ano.copy()    
    df_consumo_ppa["PPA_MW"] = df_consumo_ppa["PPA_MW"] * dias_mes
    suma_total = df_consumo_ppa["PPA_MW"].sum()
    ppa_mw = suma_total/(365*24)    

    #NPV Contrato ingreso
    df_consumo_anual = df[["ANO", "MES", "PPA_MW"]].copy()
    df_Contrato_PPA = df_consumo_anual.copy()
    df_Contrato_PPA["PPA_MENSUAL"] = df_Contrato_PPA["PPA_MW"] * dias_mes * ppa_price
    df_Contrato_PPA.drop(columns=["PPA_MW"], inplace=True)

    # NPV
    Contrato_NPV = NPV(df_Contrato_PPA,tasa_descuento_anual, anios_tot).T
    Contrato_NPV = Contrato_NPV.reset_index(drop=True)

    valor = Contrato_NPV.iloc[-1, 0]

    df_Contrato_NPV = pd.DataFrame(
        np.repeat(valor, len(df_Retiro_NPV)),
        index=df_Retiro_NPV.index
    )

    return df_Retiro_NPV, df_consumo_anual, df_stats_mes, df_Contrato_NPV, ppa_mw

#-------------------------------------------------------------------------------------------------#

def NPV(df,tasa_descuento_anual, anios_tot):

    # Parametros   
    tasa_descuento_mensual = (1 + tasa_descuento_anual) ** (1/12) - 1

    # Supongamos que ya tienes el DataFrame `df_resultado` de dimensiones (180x120)
    valores = df.values  # Convertir DataFrame en array de NumPy

    # Crear un rango de periodos desde 0 hasta 179 (filas 0 a 179)
    x = anios_tot * 12
    periodos = np.arange(x).reshape(-1, 1)  # Redimensionar para broadcasting

    # Calcular el NPV por columna
    npv_por_columna = np.sum(valores / ((1 + tasa_descuento_mensual) ** periodos), axis=0)

    # Convertir el resultado en un DataFrame (1x120)
    df_npv = pd.DataFrame([npv_por_columna], columns=df.columns)
    return df_npv


def NPV_Planta(df, MW_values,tasa_descuento_anual, factor, anios_tot):

    # Calcular el NPV para cada valor de MW
    npv_matriz = []
    i = 0
    for MW in MW_values:

        df_mw = df * MW * factor

        # NPV
        df_NPV = NPV(df_mw,tasa_descuento_anual, anios_tot)
        npv_matriz.append(df_NPV.values.flatten())

    df_NPV0 = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120
    
    return df_NPV0

def NPV_InyeccionBESS(resultados_BESS, MW_values, BESS_h_values, tasa_descuento_anual, capturado, n_hydro, anios_tot):
    
    dfs_por_h = {d["h_bess"]: d["df_ingresos"] for d in resultados_BESS}

    # Calcular el NPV para cada valor de MW
    npv_matriz = []
    i = 0
    for i, MW in enumerate(MW_values):
        h_bess = BESS_h_values[i]
        if h_bess > 0:
            df2 = dfs_por_h[h_bess]
            df = df2.loc[:, 1:n_hydro]
            df_mw = df * MW * capturado 
            df_NPV = NPV(df_mw,tasa_descuento_anual, anios_tot) # NPV
        else:
            zeros = np.zeros((1, n_hydro))
            df_NPV = pd.DataFrame(zeros)

        npv_matriz.append(df_NPV.values.flatten())


    df_NPV0 = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120

    return df_NPV0

def NPV_CosteBESS(df, MW_values, BESS_h_values, tasa_descuento_anual, coste, anios_tot):

    # Calcular el NPV para cada valor de MW
    npv_matriz = []
    i = 0
    for i, MW in enumerate(MW_values):
        h_bess = BESS_h_values[i]
        if h_bess > 0:
            coste_h = coste[f"{h_bess}h"]
            df_mw = df * MW * coste_h * h_bess
        else:
            zeros = np.zeros((anios_tot*12, 1))
            df_mw = pd.DataFrame(zeros)
        # NPV
        df_NPV = NPV(df_mw,tasa_descuento_anual, anios_tot)
        npv_matriz.append(df_NPV.values.flatten())

    df_NPV0 = pd.DataFrame(npv_matriz).T  # Transponer la lista para obtener 120
    
    return df_NPV0
