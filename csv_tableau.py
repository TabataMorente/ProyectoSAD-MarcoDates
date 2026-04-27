import pandas as pd

# ==========================================
# 1. CONSOLIDACIÓN DE CSVS
# ==========================================
# Carga tus tres archivos de clústeres
df_neg = pd.read_csv('resultados_clustering_trigrams/negativos/distribucion_docs/distribucion_docs_k6.csv')
df_neu = pd.read_csv('resultados_clustering_trigrams/neutros/distribucion_docs/distribucion_docs_k5.csv')
df_pos = pd.read_csv('resultados_clustering_trigrams/positivos/distribucion_docs/distribucion_docs_k6.csv')

# Unimos los tres archivos en uno solo (el Corpus Maestro)
df_master = pd.concat([df_neg, df_neu, df_pos], ignore_index=True)

# ==========================================
# 2. LIMPIEZA GEOGRÁFICA
# ==========================================
# Dividimos la columna 'location' usando la coma como separador.
df_master[['City', 'Country']] = df_master['location'].str.split(', ', n=1, expand=True)

# Limpiamos posibles espacios en blanco extra
df_master['City'] = df_master['City'].str.strip()
df_master['Country'] = df_master['Country'].str.strip()


# ==========================================
# 3. CREACIÓN DE CAMPOS CALCULADOS
# ==========================================
def agrupar_sentimiento(score):
    if score <= 2:
        return 'Negativo'
    elif score == 3:
        return 'Neutro'
    elif score >= 4:
        return 'Positivo'
    else:
        return 'Desconocido'

df_master['Grupo_Sentimiento'] = df_master['score'].apply(agrupar_sentimiento)


# ==========================================
# 4. MAPEO DE TÓPICOS (Nombres directos)
# ==========================================
def nombrar_clusters(row):
    # Si es Negativo (k=6)
    if row['Grupo_Sentimiento'] == 'Negativo':
        nombres_neg = {
            0: "Pagos y dinero",
            1: "Perfiles falsos y bots",
            2: "Errores y verificación",
            3: "Falta de matches",
            4: "Quejas generales",
            5: "Cuentas bloqueadas"
        }
        return nombres_neg.get(row['Cluster_asignado'], "Otros negativos")

    # Si es Neutro (k=5)
    elif row['Grupo_Sentimiento'] == 'Neutro':
        nombres_neu = {
            0: "Ver quién te dio like",
            1: "Fallos en cuentas de pago",
            2: "Cansancio de la app",
            3: "Mejoras de diseño",
            4: "Límite de mensajes gratis"
        }
        return nombres_neu.get(row['Cluster_asignado'], "Otros neutros")

    # Si es Positivo (k=6)
    elif row['Grupo_Sentimiento'] == 'Positivo':
        nombres_pos = {
            0: "Útil para viajar",
            1: "Cara pero funciona",
            2: "Fácil de usar",
            3: "Para hacer amigos",
            4: "Casos de éxito / Ayuda",
            5: "Citas rápidas"
        }
        return nombres_pos.get(row['Cluster_asignado'], "Otros positivos")

# Aplicamos la función
df_master['Nombre_Topico'] = df_master.apply(nombrar_clusters, axis=1)

# ==========================================
# 5. EXPORTAR EL ARCHIVO FINAL
# ==========================================
df_master.to_csv('Tinder_Corpus_Tableau.csv', index=False, encoding='utf-8')
print("¡Archivo maestro creado con éxito!")