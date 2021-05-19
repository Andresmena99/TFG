"""
Modulo Gestión lectura Paciente. Implementación
-----------------------------------------------

Módulo Gestión lectura Paciente

En este modulo se implementan funciones auxiliares para gestionar la lectura de 
un fichero csv con las estadísticas de un paciente.
"""
import pandas as pd
import re
import csv
import datetime

def transformar_nombres_columnas(nombres:list, return_transformer:bool =False) -> list:
    """ Realiza la transformación de los nombres de la columnas.

    A partir de una lista de nombres de columnas, transforma los mismos 
    eliminando el texto irrelevante de las mismas. 

    Args:
        nombres (:obj:`list`):
            Lista de strings que contiene los nombres de las columnas.

        return_transformer (:obj:`bool`, optional):
            Indica si se desean obtener los nuevos nombres de las columnas, o 
            si se desea obtener un transformador, diccionario, con 
            nombre_antiguo - nombre_nuevo. Defaults a False

    Returns:
        :obj:`list`: 
            Lista con la traducción de los nombres (si return_transformer es 
            False) o diccionario con {nombre_antiguo:nombre_nuevo} en caso 
            contrario.
    """
    def transformacion(nombre: str) -> str:
        # Cambiamos los nombres de las columnas para que no aparezca el timestep
        # Cambiamos los nombres de las columnas para que no aparezcan en mayuscula
        # Cambiamos los nombres de las columnas para que no haya espacios
        tmp = re.sub(' *\(Mean.*\).*', '', nombre).replace("Presi?n", "Presion").lower().replace(" ", "_")

        # Cambiamos los nombres de apnea, hipoapnea
        tmp = tmp.replace("events_", "").replace(")", "").replace("(", "").replace("[count]", "").replace("a.", "apnea").replace("h.", "hypopnea")
        tmp = traducir_castellano(tmp)
        return tmp
    
    def traducir_castellano(nombre: str) -> str:
        # Realiza la traduccion de idioma si es necesario
        translator = {"nasal_pressure":"presion_nasal", "pulse": "pulso"}
        return translator[nombre] if nombre in translator else nombre

    new_col_names = {colname: transformacion(colname) for colname in nombres} 

    return new_col_names if return_transformer else list(new_col_names.values())

def leer_csv(filename: str, time_step: float) -> pd.DataFrame:
    """Función para leer el fichero csv generado por noxturnal.

    Realiza la lectura de un fichero csv. Durante dicho proceso, realiza la 
    transformación necesaria en los nombres de las columnas, y genera las 
    medidas de tiempos necesarias con inicio las 00:00:00 para cada muestra 
    del conjunto de datos.

    Args:
        filename (:obj:`str`):  
            Nombre del fichero que contiene la información del paciente.

        time_step (:obj:`float`): 
            Tiempo entre una muestra y otra.

    Returns:
        :obj:`pd.DataFrame`: DataFrame con el contenido del fichero leído.
    
    See Also:
        :func:`transformar_nombres_columnas`, :func:`generate_time`
    """

    # En primer lugar leeemos las columnas que tiene el csv, para quitar las que estan vacías
    with open(filename, 'r') as f:
        d_reader = csv.DictReader(f)

        #get fieldnames from DictReader object and store in list
        headers = d_reader.fieldnames
    
    if '' in headers:headers.remove('')

    # No leemeos la segunda linea del fichero que indica la unidad de medida
    df_csv = pd.read_csv(filename, skiprows=[1], usecols=headers)

    # Eliminamos las filas que contengan valores vacios
    # df_csv.dropna(inplace=True)

    # Eliminamos los n primeros segundos de la muestra
    seconds_remove = 60
    df_csv.drop(index=range(0, int(seconds_remove/time_step)), inplace=True)
    df_csv.reset_index(drop=True, inplace=True)

    # Almaceno el timestamp original por si lo necesito (añadiendo los milisegundos)
    hours, minutes, seconds = [int(i) for i in df_csv["Time Stamp"][0].split(":")]
    old_timestamps = generate_time(initial_time_hour = hours, initial_time_minute = minutes, initial_time_second = seconds, time_step=time_step, number_samples=df_csv.shape[0])
    df_csv.insert(loc=1, column="Time Stamp Original", value=old_timestamps)

    # Almaceno los nuevos timestamp, empezando en el instante 0
    new_timestamps = generate_time(initial_time_hour = 0, initial_time_minute = 0, initial_time_second = 0, time_step=time_step, number_samples=df_csv.shape[0])
    df_csv["Time Stamp"] = new_timestamps
    
    # Transformamos los nombres
    new_col_names = transformar_nombres_columnas(df_csv.columns, return_transformer=True)

    df_csv = df_csv.rename(columns = new_col_names)

    return df_csv

def generate_time(initial_time_hour:int, initial_time_minute:int, initial_time_second:int, time_step:float, number_samples:int) -> list:
    """Función que genera n marcas de tiempo a partir de un instante dado.

    A partir de la hora inicial indicada en los parámetros de la función, y 
    con el espacio entre muestras indicado, genera el número de muestras 
    indicadas.

    Args:
        initial_time_hour (:obj:`int`):  
            Hora de inicio de generación de tiempos.
        initial_time_minute (:obj:`int`):  
            Minuto de inicio de generación de tiempos.
        initial_time_second (:obj:`int`):  
            Segundo de inicio de generación de tiempos.
        time_step (:obj:`int`):  
            Espacio temporal entre una muestra y la siguiente.
        number_samples (:obj:`int`):  
            Número de muestras que se deben generar.

    Returns:
        :obj:`list`: Lista con los tiempos que se han generado.
    """
    # Modificamos la columna del time stamp para que empiece en el 0
    dt = datetime.datetime(2000, 1, 1, initial_time_hour, initial_time_minute, initial_time_second)
    step = datetime.timedelta(seconds=time_step)

    result = []
    for _ in range(number_samples):
        result.append(dt.strftime('%H:%M:%S.%f')[:-3])
        dt += step
    return result


def leer_timestep(filename: str) -> float:
    """Función para leer el time_step del fichero csv generado por noxturnal.

    Utilizando las cabeceras del fichero csv generado por noxturnal, se realiza 
    la lectura de las mismas para extraer el time_step con el que se han 
    generado los datos.

    Args:
        filename (:obj:`str`): 
            Nombre del fichero que contiene la información del paciente.

    Returns:
        :obj:`float`: 
            Time step entre las muestras de las observaciones del fichero CSV.
    """
    # En primer lugar leeemos las columnas que tiene el csv, para quitar las que estan vacías
    with open(filename, 'r') as f:
        d_reader = csv.DictReader(f)
        headers = d_reader.fieldnames
    
    if '' in headers:headers.remove('')

    time_step = -1

    # Leo el time_step que tiene el fichero
    for col_name in headers:
        if (re.search("\d;\d+", col_name)):
            time_step = float(re.search("\d;\d+", col_name)[0].replace(";", "."))
            break

    if time_step == -1:
        print("Error, no se ha encontrado el timestep")
    
    return time_step

