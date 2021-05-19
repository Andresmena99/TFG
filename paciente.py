"""
Modulo Paciente. Implementación
-------------------------------

En este modulo se implementa la clase paciente, la cual se encargará 
de gestionar el procesado y cálculo de estadísticas de un paciente.

Se implementan distintas funciones para calcular los índices de apnea e 
hipopnea del paciente, así como para calcular estadísticas sobre el retraso que 
tienen las señales, y la correlación existente entre las mismas.
"""
import gestor_lectura
import patient_manager
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import pickle
import os
import utils
import numpy as np
from copy import deepcopy

class Paciente:
    """Clase encargada de la gestión de los pacientes.

    Attributes:
        df(:obj:`pd.DataFrame`): 
            Dataframe con toda la información de las señales del paciente.
        filename(:obj:`str`):  
            Nombre del fichero de donde se ha leído el paciente.
        variables_significativas(:obj:`list`): 
            Lista con la variable significativas del paciente.
        correlation_matrix(:obj:`np.matrix`): 
            Matriz de correlación de las variables significativas del
            paciente.
        sincronizacion_señales(:obj:`dict`):  
            Diccionario con clave el retraso de las señales que se han 
            estudiado, y valor DataFrame con los resultados del estudio.
        time_step(:obj:`float`): 
            Tiempo entre una muestra y otra. Calculado leyendo el nombre de las 
            columnas del fichero csv. 
        apnea_index(:obj:`float`):
            Índice de apneas por hora del paciente.
        hypopnea_index(:obj:`float`):  
            Índice de hipopneas por hora del paciente.
        iah(:obj:`float`): 
            Índice de apneas-hipopneas por hora del paciente.
    """

    df = None # Dataframe con toda la información
    filename = None # Nombre del fichero de donde se ha leído el paciente
    variables_significativas = None
    correlation_matrix = None
    sincronizacion_señales = None
    time_step = 0
    apnea_index = None 
    hypopnea_index = None 
    iah = None

    def __init__(self, filename:str, variables_significativas: list):
        """Inicializar los atributos del Paciente.

        Se lee el fichero con los datos del paciente, así como el time_step 
        entre una muestra y la siguiente que se puede extraer de los nombres de
        las columnas.

        Args:
            filename (:obj: `str`):  
                Nombre del fichero que contiene la información del paciente.

            variables_significativas (:obj: `list`): 
                Lista con las variables significativas del paciente.          
        """
        self.sincronizacion_señales = {}
        self.filename = filename
        self.time_step = gestor_lectura.leer_timestep(filename)
        self.df = \
            gestor_lectura.leer_csv(filename=filename, time_step=self.time_step)
        self.variables_significativas = \
            gestor_lectura.transformar_nombres_columnas(variables_significativas)
    
    def almacenar_paciente(self) -> None:
        """Almacena la información del paciente serializada.

        Haciendo uso de la librería Pickle, almacena el objeto paciente 
        serializado.       
        """
        pickle_dir = f"{os.path.dirname(self.filename)}/pickle"
        pickle_file = f"{pickle_dir}/{os.path.splitext(os.path.basename(self.filename))[0]}.pickle"
        utils.create_dir_if_not_exists(pickle_dir)
        with open(pickle_file, "wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def comprobar_validez_dataset(self, threshold=0.002) -> bool:
        """Comprueba que los datos del paciente sean válidado.

        Se comprueba que todas las columnas indicadas como variables 
        significativas al inicializar el Paciente contengan valores. Se tolera 
        un threshold (porcentaje) de valores nan en la columna estudiada, y 
        estos valores nan seran reemplazados por la última muestra válida de
        los datos.

        Args:
            threshold (:obj:`float`, optional):
                Porcentaje de muestras que se tolera que esté con valor nan 
                dentro de una columna. Defaults a 0.002.
        
        Returns:
            :obj:`bool`: 
                    True si el paciente se considera válido, al cumplir los 
                    requisitos del mínimo número de valores que no pueden ser 
                    nan. False en caso contrario.
        """
        for variable in self.variables_significativas:
            nan_values = self.df[variable].isnull()
            if nan_values.any(): # Si hay nan, comprobamos cuantos
                if sum(nan_values) / len(nan_values) > threshold:
                    return False
                
                # Rellenamos valor nan con la anterior observacion disponible
                self.df[variable].ffill(axis=0, inplace=True)

        return True

    def calcular_indices_paciente(self) -> None:
        """Calcula los índices de apnea, hipopnea y apnea-hipopnea.

        Calcula y almacena en los atributos del paciente los índices de apnea, 
        hipopnea y apnea-hipopnea.
        """
        self.apnea_index, self.hypopnea_index, self.iah = \
            patient_manager.calcular_indices_apnea_hypopnea(df = self.df)

    def clean_memory(self) -> None:
        """Realiza limpieza de memoria.

        Elimina la información relativa al fichero de datos que se ha leído al 
        inicializar el paciente. Función útil cuando se quieren leer múltiples 
        pacientes en RAM pero no se necesita la información leída del fichero.
        """
        if self.df is not None:
            del self.df

    def compute_correlation_matrix(self, plot_results=False, method='pearson') -> None:
        """Calcula la matriz de correlación de las variables significativas.

        Se calcula y se almacena la información de la matriz de correlación. 
        Se puede especificar el método de cálculo de la matriz de correlación, 
        y si se desea ver el gráfico del calculo
        
        Args:
            plot_results (:obj:`bool`, optional):
                Indica si se quiere mostrar el resultado gráficamente. 
                Defaults a False.
            method (:obj:`str`, optional):
                Indica el cálculo de correlación que se aplica. Los valores 
                posibles son pearson, kendall o spearman. Defaults a pearson.

        Returns:
            :obj:`pd.DataFrame`: 
                DataFrame con la matriz de correlación si el parámetro 
                plot_results es False. None en caso contrario.
        """
        if method not in ["pearson", "kendall", "spearman"]:
            print("Error. Los metodos aceptados son pearson, kendall o spearman")
            return None

        if self.correlation_matrix is None:
            self.correlation_matrix = \
                self.df[self.variables_significativas].corr(method=method)

        if plot_results is False: return self.correlation_matrix

        # Establecemos el tamaño del mapa de calor
        plt.figure(figsize=(13, 13))

        # Indicamos que los valores del mapa de calor sean entre -1 y 1
        # annotation = True para que se muestren los valores del mapa de calor.
        heatmap = sns.heatmap(self.correlation_matrix, 
                                cmap='coolwarm', vmin=-1, vmax=1, annot=True)

        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        # heatmap.set_xticklabels(rotation=90)
        # heatmap.set_yticklabels(rotation=270)
        # Give a title to the heatmap. Pad defines the distance of the title from the top of the heatmap.
        plt.title('Mapa de calor correlación');
        plt.show()

    def compute_sincronizacion_señales(self, seconds_prior = 60, seconds_post = 10, plot_results=False, signals = None, main_signal = "spo2") -> None:
        """Calcula el retraso entre dos señales.

        Para realizar el cálculo de la sincronización (el retraso) entre dos 
        señales, se hace uso del módulo patient manager. Dicho retraso consiste 
        en encontrar el desplazamiento necesario a aplicar a una señal frente 
        a la otra para alcanzar el pico de correlación. 
        
        Args:
            seconds_prior (:obj:`float`, optional):
                Segundos previos sobre los que se va a desplazar la muestra 
                para calcular la correlación. Defaults a 60.
            seconds_post (:obj:`float`, optional):
                Segundos posteriores sobre los que se va a desplazar la muestra 
                para calcular la correlación. Defaults a 10.
            plot_results (:obj:`bool`, optional):
                Indica si se quiere mostrar el resultado gráficamente. 
                Defaults a False.
            signals (:obj:`list`, optional):
                Lista de señales que se desea comparar contra la señal 
                main_signal. Defaults a None, en cuyo caso se usan todas las 
                señales significativas del paciente.
            main_signal (:obj:`str`, optional):
                Señal usada como referencia contra las que se comparan todas 
                las señales del parámetro signals.

        See Also:
            :func:`patient_manager.plot_sincronizacion_temporal_simple`.
        """
        if signals is None: signals = self.variables_significativas

        # Compruebo si ya he calculado las estadísticas previamente
        clave_diccionario = f"signal-vs-{main_signal}-prior={seconds_prior}s-post={seconds_post}s" 
        if clave_diccionario not in self.sincronizacion_señales.keys(): # Si no hemos calculado todavía las estadísticas, las calculamos
            df = pd.DataFrame(index = signals, columns=["peak_corr", "offset_peak_corr", "rs"])
            for variable in signals:
                    rs, lag_list = patient_manager.sincronizacion_temporal_simple(self.df, time_step=self.time_step, column1 = variable, column2 = main_signal, seconds_prior=seconds_prior, seconds_post=seconds_post, plot_results=False)
                    
                    if plot_results:
                        patient_manager.plot_sincronizacion_temporal_simple(rs=rs, lag_list=lag_list, time_step=self.time_step, column1 = variable, column2 = main_signal, seconds_prior=seconds_prior, seconds_post=seconds_post)

                    _,peak_correlation,_,_,offset_seconds = patient_manager.extract_stats_from_rs(rs=rs, seconds_prior=seconds_prior, seconds_post=seconds_post, time_step=self.time_step)

                    df["peak_corr"][variable] = peak_correlation
                    df["offset_peak_corr"][variable] = offset_seconds
                    df["rs"][variable] = rs

            self.sincronizacion_señales[clave_diccionario] = df
        
        # Si las estadísticas ya estaban calculadas, miro si tengo que mostrar los resultados
        elif plot_results:
            lag_list = list(range(-int(seconds_prior/self.time_step),int(seconds_post/self.time_step+1)))
            for variable in signals:
                rs = self.sincronizacion_señales[clave_diccionario]["rs"][variable]
                patient_manager.plot_sincronizacion_temporal_simple(rs=rs, lag_list=lag_list, time_step=self.time_step, column1 = variable, column2 = main_signal, seconds_prior=seconds_prior, seconds_post=seconds_post)

    def eliminar_outliers(self, df, column, max_deviations=5) -> pd.DataFrame:
        """Elimina los outliers de una columna determinada de un dataframe.

        En función de la desviación máxima índica, se eliminan los outliers 
        sobre una copia del dataframe indicado como parámetro, en la columna 
        indicada.
        
        Args:
            df (:obj:`pd.DataFrame`):
                Dataframe que contiene la columna sobre la que se desean 
                eliminar los outliers.
            column (:obj:`str`):
                Nombre de la columna del dataframe de la que se desean eliminar
                los outliers.
            max_deviations (:obj:`int`, optional):
                Desviación a partir de la cual se considera que los valores son 
                valores atípicos. Defaults a 5.

        Returns:
            :obj:`pd.DataFrame`: 
                Copia del DataFrame proporcionado como parámetro con la columna 
                indicada sin outliers.

        See Also:
            :func:`suavizar_seniales`.
        """
        an_array = df[column]
        mean = np.mean(an_array)
        standard_deviation = np.std(an_array)
        distance_from_mean = abs(an_array - mean)
        not_outlier = distance_from_mean < max_deviations * standard_deviation
        not_ourlier_index = [i for i, no_outlier in enumerate(not_outlier) if no_outlier == True]
        return deepcopy(df.iloc[not_ourlier_index])

    def suavizar_seniales(self, smooth_seconds = 5, columns = ["spo2", "pulso"]) -> None:
        """Realiza el suavizado de la señal utilizando una ventana desalizante.

        Se debe indicar el tamaño de la ventana deslizante, así como las 
        columnas sobre las que se desea aplicar el suavizado de la señal. De 
        igual forma, se eliminan los outliers de las señales que se suavizan.
        El DataFrame con las señales suavizadas se almacena dentro de la 
        clase, en el atributo df_smoothed.
        
        Args:
            smooth_seconds (:obj:`int`, optional):
                Numero de segundos que forman la ventana deslizante del 
                suavizado. Defaults a 5.
            columns (:obj:`list`):
                Lista de nombres de columnas del dataframe sobre las que se 
                desea eliminar los outliers. Defaults a ["spo2", "pulso"]

        See Also:
            :func:`eliminar_outliers`.
        """
        # En primer lugar eliminamos los outliers de la columna
        for column in columns:
            self.df_smothed = self.eliminar_outliers(self.df, column=column) 
            self.df_smothed = self.eliminar_outliers(self.df, column=column)

        def moving_average(data, window_size) :
            if isinstance(data, pd.core.series.Series):
                data = data.values

            ret = np.cumsum(data, dtype=float)
            ret[window_size:] = ret[window_size:] - ret[:-window_size]
            return ret[window_size - 1:] / window_size

        def aux_function(colnames: list, numeros_segundo_media: list):
            new_samples = []
            for colname, numero_segundo_media in zip (colnames, numeros_segundo_media):
                n = round(numero_segundo_media/self.time_step)
                moving = moving_average(self.df_smothed[colname], window_size=n)
                new_vector = np.concatenate((self.df_smothed[colname][:numero_segundo_media], moving))
                new_samples.append(new_vector)
            
            # Cojo el vector que tenga menor longitud
            min_length = min([len(i) for i in new_samples])

            # Tengo que tirar las n ultimas muestras
            n = self.df_smothed.shape[0] - min_length

            self.df_smothed.drop(self.df_smothed.tail(n).index,inplace=True) # drop last n rows
          
            # Inserto las nuevas muestras con smooth
            for colname, new_vect in zip (colnames, new_samples):
                self.df_smothed[colname] = new_vect[: min_length]
                self.df_smothed[colname] = new_vect[: min_length]

        # Realizamos el suavizado de las variables
        if smooth_seconds > 1: 
            aux_function(colnames=columns, numeros_segundo_media=[smooth_seconds, smooth_seconds-1])

    def __str__(self):
        """Devuelve representación en forma de texto de un Paciente.

        En la representación en formato de texto, se imprimen los indices de 
        apnea, hipopnea y apnea-hipopnea, así como las estadísticas de retraso 
        calculadas.
        """
        str_paciente = f"Paciente del fichero: {self.filename}\n"

        if self.apnea_index is not None:
            str_paciente += "\nÍndices paciente:\n"
            str_paciente += f"\tÍndice Apnea: {self.apnea_index:.2f}\n"
            str_paciente += f"\tÍndice Hypopnea: {self.hypopnea_index:.2f}\n"
            str_paciente += f"\tÍndice Apnea-hypopnea: {self.iah:.2f}\n"

        # for key, df in self.sincronizacion_señales.items():
        #     str_paciente += "\nEstadísticas retraso paciente:\n"
        #     str_paciente += str(key) + "\n"
        #     str_paciente += tabulate(df.drop("rs", axis=1), headers='keys', tablefmt='psql')

        return str_paciente

    def __repr__(self):
        """Devuelve la representación de un objeto de tipo Paciente.
        """
        return self.__str__()
