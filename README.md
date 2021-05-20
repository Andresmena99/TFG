# TFG (estudio de apnea)

Repositorio con el código utilizado para el desarrollo del TFG.

La **documentación** del programa para el cálculo de estadísticas de los pacientes, se encuentra disponible en [este enlace](https://andresmena99.github.io/TFG/). En dicha documentación encontraremos los detalles de funcionalidad de los siguientes módulos:

* gestor_lectura.py
* paciente.py
* patient_manager.py
* utils.py

Por protección de datos, el conjunto de datos empleado en el TFG no se encuentra disponible por el momento. Para descargar los modelos que se han entrenado y expuesto como parte de los resultados en algunos notebooks, se pueden obtener de [este enlace](https://drive.google.com/drive/folders/1_p3r5Xlm0UgG72LUH3eumQnSHfzjsuDF?usp=sharing). Los directorios descargados deberán situarse en la raíz del proyecto.

Para la visualización de los resultados obtenidos, así como el código fuente, se recomienda seguir el siguiente orden:

1. lectura_paciente_edf.ipynb
2. lectura_paciente_csv.ipynb
3. preprocesado_señal.ipynb
4. calculo_correlacion_y_retraso_pacientes.py
5. estudio_correlacion_variables.ipynb
6. estudio_retrasos_completo.ipynb
7. entrenamiento_y_visualizacion_mlp.ipynb
8. random_search_mlp.py
9. resultados_random_search_mlp.ipynb
10. generacion_dataset_lstm.ipynb
11. entrenamiento_y_visualizacion_lstm.ipynb
