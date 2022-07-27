# ¿Cómo obtener métricas del contenedor?
El contenedor debe estar corriendo antes de seguir los siguientes pasos.

## Proceso de medición

### 1. ¿Cómo registrar el uso de GPU?

Correr desde una terminal el siguiente comando:

```
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f GPU_TEST.csv
```

Esto generará el archivo de salida: `GPU_TEST.csv`



### 2. ¿Cómo registrar el uso de CPU y RAM?

Correr desde una terminal el siguiente comando:

```
while true; do docker stats --no-stream | tee --append stats.txt; sleep 1; done
```

Esto generará el archivo de salida: `stats.txt`


### 3. ¿Cómo iniciar los trials?

Se incluye entre los archivos de este directorio, `benchclient.py`, el cual permite generar _trials_ que corresponden a un número N de requests generadas por N clientes. Cada cliente es generado como una thread distinta, que se conecta con el servidor. Para cada _trial_ se genera un número aleatorio de clientes, entre el número mínimo (1) y el número máximo fijado en los argumentos (N). Para utilizar el script de benchmarking:

```
python3 benchclient.py -trials M -users N
```

En este caso, se generarán M trials, donde cada trial tendrá un número aleatorio entre 1 y N clientes.

## Análisis de resultados:

Para realizar el análisis de resultados se implementa el script `plotresults.py`, el cual permite generar gráficos de utilización para los archivos obtenidos del proceso de medición. Para utilizarlo se puede utilizar el siguiente comando:

```
python3 plotresults.py --stats stats.txt --gpu GPU_TEST.csv --show --extra
```
Donde los argumentos pasados corresponden a:

- `--stats stats.txt`: Corresponde al archivo de uso de CPU y RAM.
- `--gpu GPU_TEST.csv`: Corresponde al archivo de uso de GPU.
- `--show` : Indica que se muestra el resultado por pantalla.
- `--extra`: Realiza el análisis de promedio y desviación estándar para el uso de RAM. 
