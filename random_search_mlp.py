"""
Entrenamiento perceptron multicapa 
----------------------------------

Este programa se encarga de aplicar el algoritmo random search para entrenar 
multiples configuraciones de hiperparámetros del perceptron multicapa. El 
algoritmo se aplica usando tantos procesadores como se indique por argumento, 
ejecutando en cada procesador una instancia del programa "entrenamiento_mlp.py"
"""
import time
import argparse
import numpy as np
import multiprocessing
import subprocess

def ejecutar_entrenamientos(n_execution, batch_size, neuronas_ocultas):
    """Realiza el entrenamiento de un MLP.

    Utilizando los parámetros de la función, ejecuta el programa 
    "entrenamiento_mlp.py".  
    
    Args:
        n_execution (:obj:`int`):
            Corresponde con el identificador que se le ha asignado a la 
            ejecución actual.

        batch_size (:obj:`int`):
            Indica el batch_size de la red que se va a entrenar.
        
        neuronas_ocultas (:obj:`lista`):
            Lista con el número de neuronas ocultas en cada una de las capas
            ocultas.
    """
    print(f"Ejecutando red. Porcentaje completado: {n_execution}:\n\tneuronas_ocultas: {neuronas_ocultas}\n\tbatch_size:{batch_size}\n")
    ini = time.time()
    command = f"python3 entrenamiento_mlp.py --neuronas_ocultas {' '.join(str(i) for i in neuronas_ocultas)} --batch_size {batch_size} --seconds_prior 30 --seconds_post 10 --smooth_seconds 0"
    print(command)
    subprocess.run(command.split(), stdout=subprocess.DEVNULL) # Ejecutamos sin leer la salida
    print(f"Ejecucion red. Porcentaje completado: {n_execution}. Terminado en {time.time() - ini}s\n")

if __name__ == "__main__":
    """Realiza el entrenamiento del múltiples perceptrones multicapa.

    Haciendo uso de la librería multiprocessing, se entrenan tantos 
    perceptrones multicapas como se indique en la sección de variables 
    configurables disponible en el código, utilizando el número de procesadores 
    que se indique como argumento al programa.

    See Also:
        ejecutar_entrenamientos().
    """
    max_procesors = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--CPUs", type=int, choices=[i+1 for i in range(max_procesors)], required=True,
                        help="Número de procesadores a utilizar")

    args = parser.parse_args()

    n_processors = args.CPUs

    """"""""""""""""""""""""""""""""
    """"Variables configurables"""""
    """"""""""""""""""""""""""""""""
    n_muestras = 15 # Indica cuantas ejecuciones vamos a realizar
    min_neuronas_capa = 100 # Numero minimo de neuronas en capa oculta
    max_neuronas_capa = 2000 # Numero maximo de neuronas en capa oculta
    multiplicidad_neuronas = 50 # Indica que el numero de neuronas tiene que se multiplo de 50
    min_batch_size = 32
    max_batch_size = 128
    multiplicidad_batch_size = 32 # Indica que el batch_size de 32
    max_capas_ocultas = 2 # Limite de capas ocultas a utilizar

    n_capas = np.random.randint(low=1, high=max_capas_ocultas+1, size=n_muestras) # Capas ocultas de la muestra n

    n_neuronas = [] # Queremos evitar configuraciones iguales de neuronas
    for capas in n_capas: # Neuronas de cada capa de la muestra n
        # Decidimos que el numero de neuronas sea multiplo de 50
        nuevo_conjunto_neuronas = np.random.choice(range(min_neuronas_capa, max_neuronas_capa+1, multiplicidad_neuronas), capas)
        n_neuronas.append(nuevo_conjunto_neuronas) # Neuronas de cada capa de la muestra n

    batch_sizes = np.random.choice(range(min_batch_size, max_batch_size+1, multiplicidad_batch_size), n_muestras)

    params = zip(range(1, n_muestras+1),batch_sizes, n_neuronas)
    with multiprocessing.Pool(n_processors) as p:
        p.starmap(ejecutar_entrenamientos, params)