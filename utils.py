"""
Modulo Utils. Implementación
----------------------------

Modulo con funciones auxiliares

En este modulo se implementan funciones auxiliares útiles para el desarrollo 
del proyecto
"""
import os
import shutil
import numpy as np
import errno
from matplotlib.colors import ListedColormap
from matplotlib import cm

def create_dir_if_not_exists(dirname) -> None:
    """Crea un directorio en caso de que el mismo no exista.

    Args:
        dirname (:obj:`str`):
            Ruta del directorio que se quiere crear.
    """
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
            
def copy_datasets_to_project(origin: str, dest: str) -> None:
    """Mueve todos los ficheros csv del origen al destino.

    El origen es una directorio con directorios (uno por paciente), en los que 
    se encuentran los ficheros csv con la información del mismo. El destino es  
    es el directorio donde se copiaran todos estos ficheros csv.

    Args:
        origin (:obj:`str`):
            Directorio de origen donde se encuentran todos los pacientes que 
            se desean copiar.

        dest (:obj:`str`):
            Ruta donde se van a copiar todos los datasets.
    """
    create_dir_if_not_exists(dest)
    for paciente_f in os.listdir(origin):
        paciente_actual_dir = os.path.join(origin, paciente_f)
        if os.path.isdir(paciente_actual_dir):
            if "data.csv" in os.listdir(paciente_actual_dir):
                shutil.copyfile(os.path.join(paciente_actual_dir, "data.csv"), 
                                os.path.join(dest, f"data_{paciente_f}.csv"))

def set_font_size(plt, size:int=0) -> None:
    """Modifica el tamaño de los textos de la librería matplotlib.

    Dado un modulo matplotlib.pyplot, se modifica el tamaño de la fuente del 
    mismo.

    Args:
        plt (:obj:`module`):
            Modulo matplotlib.pyplot en el que se van a cambiar el tamaño de 
            fuente.

        size (:obj:`int`):
            Tamaño de fuente que se quiere establecer.
    """
    params = {f'legend.fontsize': size,
         f'axes.labelsize':  size,
         f'axes.titlesize': size+4,
         f'xtick.labelsize': size,
         f'ytick.labelsize': size,
         f'font.size': size}

    plt.rcParams.update(params)

def color_map() -> ListedColormap:
    """Genera un mapa de color personalizado tomando como base el 'copper'.

    Tomando como base el color copper de matplotlib, se genera un mapa 
    personalizado en el que se eliminan los tonos más oscuros del mismo.

    Returns:
        :obj:`ListedColormap`: Mapa de color personalizado.
    """
    copper = cm.get_cmap('copper', 512)
    newcmp = ListedColormap(copper(np.linspace(0.2, 1, 256)))
    return newcmp
