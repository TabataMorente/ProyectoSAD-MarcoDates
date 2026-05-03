# 🚀 SAD Proyecto Grupal

## 🐍 Requisitos de Software
* **Versión de Python:** `3.12`.

---

## 💻 Cómo ejecutar el proyecto (Llamadas)

Para ejecutar el flujo completo (entrenamiento, selección del mejor modelo, evaluación final y generación de matrices de confusión), utiliza el script `main.py` con los siguientes argumentos:

```bash
python main.py tinder.csv config_file.json
```

### Cómo ejecutar el archivo de prompt_engineering.py

#### Clasificación
En caso de querer hacer la clasficación, se usan los mismos parámetros
al ejecutar el script.
```bash
python langChain.py tinder.csv config_file.json
```
> El **resultado** se guarda en `classification_results` y contiene
> archivos **divididos por shots**.
#### Paráfrasis
Para poder hacer oversampling generativo la ejecución del script es diferente.
```bash
python langChain.py tinder.csv config_file.json parafrasis.csv
```
El archivo parafrasis.csv contiene dos columnas. La primera `id` hace referencia
al número de fila de la instancia que se encuentra en el archivo `tinder.csv`. La
segunda columna `ejemplos` contiene ejemplos de qué sería una paráfrasis al comentario
que se encuentra en `tinder.csv`.
> Si se quiere poner **más de un ejemplo** para un mismo comentario,
> hay que crear una nueva fila que haga referencia al mismo ID.
> Para **no poner ejemplos** en un comentario, pero se quiere hacer parafrásis
> con este, deja en esa fila la columna `ejemplos` vacía.

Este es un **ejemplo**:
```csv
id, ejemplo
1, I am not able to delete my account, so could you do it for me
1, Can you delete my account, I can't do it myself
2,
```

> El **resultado** se guarda en `oversample_results` dividido por shots
> y como un único archivo llamado `Tinder_oversample.csv`

##### Oversampling generativo
El script de Python `join_datasets.py` se usa para unir el dataset original
con los resultados generados por el modelo de lenguaje en un nuevo csv,
que se llama `new_Tinder.csv` en la carpeta que el usuario considere.
Esta sería la forma de ejecutar el script:
```bash
python join_datasets.py directorio_parafrasis_generados tinder.csv directorio_resultado
```
> **No hace falta** crear `directorio_resultado` antes de ejecutar el script.
