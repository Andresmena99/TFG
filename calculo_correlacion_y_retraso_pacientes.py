"""
Calculo correlacion y retraso multiprocesador
---------------------------------------------

Este programa se encarga de realizar la lectura de los pacientes disponibles en
un directorio dado. De estos pacientes se calcula los índices de apnea, 
hipopnea y apnea-hipopnea, así como la matriz de correlación de las señales de 
la poligrafía nocturna y el retraso de las mismas. Una vez calculada esta
información, se almacena serializada.
"""

import time
import argparse
import multiprocessing
import os
from paciente import Paciente

def procesar_paciente(n_execution, dataset_dir):
    """Realiza el procesamiento de un paciente.

    A partir del fichero donde se encuenta el paciente que se desea leer, se 
    calculan los índices de apnea, hipopnea y apnea-hipopnea, así como la 
    matriz de correlación de las señales de la poligrafía nocturna y el 
    retraso de las mismas. Una vez calculada esta información, se almacena el 
    paciente serializado usando la librería Pickle.

    Args:
        n_execution (:obj:`int`):
            Corresponde con el identificador que se le ha asignado a la 
            ejecución actual.

        dataset_dir (:obj:`str`):
            Corresponde con el path del fichero que contiene el conjunto de 
            datos del paciente. Dicho fichero es un CSV que ha sido exportado 
            usando la herramienta Noxturnal.
    """
    # Indicamos las variables que vamos a querer estudiar. Tienen que ser 
    # variables que se encuentren disponibles en el fichero csv
    variables_significativas = ['Abdomen','Activity', 'Audio Volume dB', \
                            'Flow', 'Presion Nasal', 'Pulso','SpO2','Snore']

    print(f"{time.strftime('%H:%M')}\tProcesando paciente # {n_execution}:" + \
                                            f"\n\tDataset: {dataset_dir}\n")
    ini = time.time()

    # Realizamos la lectura del paciente, calculamos los indices, correlacion 
    # y retraso, y se almacena el paciente
    paciente = Paciente(filename=dataset_dir, 
                            variables_significativas=variables_significativas)

    # Compruebo que el dataset me sirva, sino lo ignoro
    if paciente.comprobar_validez_dataset(): 

        # Calculo indices del paciente
        paciente.calcular_indices_paciente() 
        for signal in paciente.variables_significativas:
            # La señal de saturación sabemos que tiene que estar adelantada, 
            # por lo que su retraso lo calculamos de forma especia
            if signal != "spo2":
                paciente.compute_sincronizacion_señales(main_signal=signal, 
                                        seconds_prior = 60, seconds_post = 60)
            else:
                paciente.compute_sincronizacion_señales(main_signal=signal, 
                                        seconds_prior = 60, seconds_post = 10)

        # Calculo la matriz de correlacion
        paciente.compute_correlation_matrix() 

        # Se almacena el paciente serializado
        paciente.almacenar_paciente()

    else: 
        print(f"Paciente no válido:")
        print(paciente)

    print(f"{time.strftime('%H:%M')}\tFinalizado paciente # {n_execution}." + \
                                        f"Terminado en {time.time() - ini}s\n")

if __name__ == "__main__":
    """Realiza el procesamiento de todos los paciente.

    Haciendo uso de la librería multiprocessing, se leen todos los pacientes 
    disponibles dentro de un directorio, utilizando el número de procesadores 
    que se indique como argumento al programa.

    See Also:
        procesar_paciente().
    """
    
    # Obtenemos el número de procesadores disponibles en el equipo
    max_procesors = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--CPUs", type=int, 
                    choices=[i+1 for i in range(max_procesors)], required=True,
                    help="Número de procesadores a utilizar")

    args = parser.parse_args()

    n_processors = args.CPUs
    """"""""""""""""""""""""""""""""
    """"Variables configurables"""""
    """"""""""""""""""""""""""""""""
    # Directorio que contiene los ficheros .csv que se van a analizar
    dataset_dir = "../data/pacientes/big-data"

    # Miramos todos los archivos dentro del directorio que tienen extension 
    # CSV, correspondientes con los archivos de los pacientes.
    pacientes_dir = [os.path.join(dataset_dir, f) 
          for f in os.listdir(dataset_dir) if os.path.splitext(f)[1] == ".csv"]

    inicial = time.time()
    params = zip([i for i in range(len(pacientes_dir))], pacientes_dir)
    with multiprocessing.Pool(n_processors) as p:
        p.starmap(procesar_paciente, params)