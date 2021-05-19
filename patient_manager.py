"""
Modulo Manager Paciente. Implementación
---------------------------------------

En este modulo se implementan funciones auxiliares para la gestión de las 
estadísticas de un paciente.

Se implementan las funciones de calculo de índices de apnea, hipopnea e 
apnea-hipopnea, así como las funciones para calcular estadísticas sobre el 
retraso que tienen las señales.
"""
import pandas as pd
import numpy as np
import gestor_lectura
import matplotlib.pyplot as plt

def obtener_tramos_activos(df: pd.DataFrame, colname: str) -> list:
    """ Calcula los índices de los tramos activos.

    Dado un dataframe, y un nombre de columna a estudiar, se analiza en qué 
    tramos el valor de la columna indicada se encuentra 1. 

    Args:
        df (:obj:`pd.DataFrame`):
            DataFrame que contiene la información que se desea analizar.

        colname (:obj:`str`):
            Nombre de la columna del DataFrame que se desea estudiar. 

    Returns:
        :obj:`list`: Lista de tuplas con los tramos activos.
    """
    tramos = np.where(df[colname] == 1)[0]
    tramos_reales = []
    if len(tramos) > 0:
        inicial = tramos[0]
        for prev, post in zip(tramos, tramos[1:]):
            if prev != post - 1:
                tramos_reales.append((inicial, prev))
                inicial = post
        tramos_reales.append((inicial, tramos[-1]))
        
    return tramos_reales

def calcular_indices_apnea_hypopnea(df: pd.DataFrame) -> list:
    """Calcula los índices apnea, hipopnea y apnea-hipopnea.

    Dado un dataframe con los datos de un paciente, se estudian los índices. 

    Args:
        df (:obj:`pd.DataFrame`):
            DataFrame que contiene la información de un paciente.

    Returns:
        :obj:`list`: 
            La lista contiene los índices apnea, hipopnea y apnea-hipopnea.

    See Also:
        :func:`obtener_tramos_activos`.
    """
    def calcular_indice(df: pd.DataFrame, colnames: list):
        tramos_activos = [obtener_tramos_activos(df, col) for col in \
                        gestor_lectura.transformar_nombres_columnas(colnames)]

        # Calculamos el numero de tramos activos
        num_tramos_activos = sum([len(i) for i in tramos_activos])

        # Calculamos cuantas horas ha durado la prueba
        hours, minutes, seconds = df["time_stamp"].iloc[-1].split(":")
        duracion_prueba_horas = int(hours) + int(minutes)/60 + float(seconds)/3600

        # Calculamos el indice
        return num_tramos_activos/duracion_prueba_horas

    # Columnas que indican tramos de apnea
    apnea_columns = ["apnea obstructive", "apnea central", "apnea mixed"]
    apnea_index = calcular_indice(df, apnea_columns)

    # Columnas que indican tramos de hypoapnea
    hypopnea_columns = ["hypopnea central", "hypopnea obstructive", "hypopnea"]
    hypopnea_index = calcular_indice(df, hypopnea_columns)

    return apnea_index, hypopnea_index, apnea_index+hypopnea_index

def extract_stats_from_rs(rs, seconds_prior, seconds_post, time_step) -> list:
    """Extrae las estadísticas a partir de una lista con las correlaciones.

    A partir de la lista de correlaciones, e indicando los segundos previos y 
    posteriores, se calcula en qué posición se encuentra la máxima correlación,
    así como con qué retraso.

    Args:
        rs (:obj:`list`):
            lista de valores de la correlación
        time_step (:obj:`float`):
                time_step con el espacio entre una muestra y la siguiente.
        seconds_prior (:obj:`float`):
            Segundos previos sobre los que se va a desplazar la muestra para 
            calcular la correlación.
        seconds_post (:obj:`float`):
            Segundos posteriores sobre los que se va a desplazar la muestra 
            para calcular la correlación.

    Returns:
        :obj:`list`: 
            Devuelve una lista con cinco elementos, donde cada uno indica: 
            Posición del pico de correlación, pico de correlación, posición de 
            la correlación sin desplazamiento, correlación sin desplazamiento, 
            y el retraso de una señal respecto a la otra.  

    See Also:
        :func:`sincronizacion_temporal_simple`. 
    """
    # Miramos cual es el mayor valor de correlación (en valor abosluto). 
    peak_correlation_pos = np.argmax(np.abs(rs))
    peak_correlation = rs[peak_correlation_pos]

    # Miramos cual es el valor de la correlación sin desplazar la señal
    # Para ello, tenemos que tener en cuenta la proporcion de segundos anterior y posteriores, para encontrar
    # donde estaba la señal centrada
    proporcion_prior_post = seconds_prior / (seconds_post+seconds_prior) if seconds_prior > seconds_post else seconds_post / (seconds_post+seconds_prior)
    posicion_centro = int(len(rs)*proporcion_prior_post)
    correlacion_centro = rs[posicion_centro]

    # Calculamos el retraso de la señal spo2
    offset = np.floor(posicion_centro-peak_correlation_pos)
    offset_seconds = offset * time_step

    return peak_correlation_pos,peak_correlation,posicion_centro,correlacion_centro,offset_seconds

def plot_sincronizacion_temporal_simple(rs, lag_list, column1, column2, time_step, seconds_prior, seconds_post, savefig=False) -> None:
    """Realiza la representación gráfica del retraso entre dos señales.

    Se representa gráficamente el retraso. Utilizando los parámetros de la 
    función, se especifican los resultados que se han obtenido al calcular la 
    correlación conforme se desplaza la señal. 

    Args:
        rs (:obj:`list`):
            lista de valores de la correlación.
        lag_list (:obj:`list`):
            lista con los desplazamientos en los que se ha calculado la 
            correlación.
        column1 (:obj:`str`):
            Primera columna de la que se quieren extraer los datos para 
            calcular la sincronización.
        column2 (:obj:`str`):
            Segunda columna de la que se quieren extraer los datos para 
            calcular la sincronización.
        time_step (:obj:`float`):
            time_step con el espacio entre una muestra y la siguiente.
        seconds_prior (:obj:`float`):
            Segundos previos sobre los que se va a desplazar la muestra para 
            calcular la correlación.
        seconds_post (:obj:`float`):
            Segundos posteriores sobre los que se va a desplazar la muestra 
            para calcular la correlación.
        savefig (:obj:`bool`, optional):
            Indica si se quiere almacenar el gráfico del resultado en un 
            fichero pdf. Defaults a False.

    See Also:
        :func:`sincronizacion_temporal_simple`, 
        :func:`extract_stats_from_rs`.
    """
    peak_correlation_pos,peak_correlation,posicion_centro,correlacion_centro,offset_seconds = extract_stats_from_rs(rs=rs, seconds_prior=seconds_prior, seconds_post=seconds_post, time_step=time_step)

    f,ax=plt.subplots(figsize=(10, 5))
    ax.plot(rs, color="C0")
    ax.axvline(posicion_centro,color='k',linestyle='--',label=f'Center. Corr = {correlacion_centro:.3f}')
    ax.axvline(peak_correlation_pos,color='C1',linestyle='--',label=f'Peak correlation. Corr = {peak_correlation:.3f}')
    ax.set(title=f'Retraso señal {column2} frente a {column1}\nOffset = {offset_seconds:.2f} segundos', xlabel='Offset (segundos)',ylabel='Pearson r')

    # Ajustamos el eje horizontal para que muestre las unidades en segundos de adelanto-retraso
    # Representamos un total de 10 ticks
    total_ticks = 10
    tick_step = int(len(rs)/total_ticks)
    xticks_pos = [i*tick_step for i in range(total_ticks+1)]
    xticks_labels = [round(i*time_step, 2) for i in lag_list[0::tick_step]]

    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_labels)
    
    plt.legend()
    if savefig: 
        plt.savefig("plot_sincronizacion_temporal_simple.pdf", transparent=True, bbox_inches = 'tight',pad_inches = 0.025)
    plt.show()


def sincronizacion_temporal_simple(df, time_step, column1, column2, seconds_prior = 10, seconds_post = 10, plot_results = True) -> list:
    """Calcula el retraso de una señal frente a otra.

    Se calcula el retraso de una señal frente a otra. Utilizando los parámetros 
    de la función, se puede especificar el rango de desplazamiento de las 
    señales, así como cuánto se ha de desplazar cada una de las señales. 

    Args:
        df (:obj:`pd.DataFrame`):
            DataFrame que contiene la información de un paciente.
        time_step (:obj:`float`):
            time_step con el espacio entre una muestra y la siguiente.
        column1 (:obj:`str`):
            Primera columna de la que se quieren extraer los datos para 
            calcular la sincronización.
        column2 (:obj:`str`):
            Segunda columna de la que se quieren extraer los datos para 
            calcular la sincronización.
        seconds_prior (:obj:`float`, optional):
            Segundos previos sobre los que se va a desplazar la muestra para 
            calcular la correlación. Defaults a 10.
        seconds_post (:obj:`float`, optional):
            Segundos posteriores sobre los que se va a desplazar la muestra 
            para calcular la correlación. Defaults a 10.
        plot_results (:obj:`bool`, optional):
            Indica si se quiere mostrar el resultado gráficamente. 
            Defaults a False.

    Returns:
        :obj:`list`: 
            Si se muestran los resultados gráficamente, no devuelve nada. 
            En caso contrario, devuelve una lista con el primer elemento una 
            lista de valores de la correlación, y el segundo elemento una  
            lista con los desplazamientos en los que se ha calculado la 
            correlación.

    See Also:
        :func:`plot_sincronizacion_temporal_simple`.
    """
    # Extraemos las variables que vamos a estudiar
    data_variable = df[column1]
    data_spo2 = df[column2]

    # Lista con los valores de offset hacia delante y hacia atrás
    lag_list = list(range(-int(seconds_prior/time_step),int(seconds_post/time_step+1)))

    # Calcularamos la correlación con todos los desplazamientos
    rs = [data_variable.corr(data_spo2.shift(lag)) for lag in lag_list]

    # Realizamos gráfica con resultados, o devolvemos los cálculos de las correlaciones
    if plot_results:
       plot_sincronizacion_temporal_simple(rs=rs, lag_list=lag_list, column1=column1, column2=column2, time_step=time_step, seconds_prior=seconds_prior, seconds_post=seconds_post)
    else:
        return rs, lag_list