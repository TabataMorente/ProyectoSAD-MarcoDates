import pandas as pd

# ==========================================
# 1. CARGA Y RENOMBRADO DE COLUMNAS (Soft Clustering)
# ==========================================
# Diccionarios con los nombres exactos para las columnas de probabilidad
nombres_neg = {
    'cluster_0': "Prob Neg: Pagos y dinero",
    'cluster_1': "Prob Neg: Perfiles falsos y bots",
    'cluster_2': "Prob Neg: Errores y verificación",
    'cluster_3': "Prob Neg: Falta de matches",
    'cluster_4': "Prob Neg: Quejas generales",
    'cluster_5': "Prob Neg: Cuentas bloqueadas"
}

nombres_neu = {
    'cluster_0': "Prob Neu: Ver quién te dio like",
    'cluster_1': "Prob Neu: Fallos en cuentas de pago",
    'cluster_2': "Prob Neu: Cansancio de la app",
    'cluster_3': "Prob Neu: Mejoras de diseño",
    'cluster_4': "Prob Neu: Límite de mensajes gratis"
}

nombres_pos = {
    'cluster_0': "Prob Pos: Útil para viajar",
    'cluster_1': "Prob Pos: Cara pero funciona",
    'cluster_2': "Prob Pos: Fácil de usar",
    'cluster_3': "Prob Pos: Para hacer amigos",
    'cluster_4': "Prob Pos: Casos de éxito / Ayuda",
    'cluster_5': "Prob Pos: Citas rápidas"
}

# Cargamos los archivos
df_neg = pd.read_csv('resultados_clustering_trigrams/negativos/distribucion_docs/distribucion_docs_k6.csv')
df_neu = pd.read_csv('resultados_clustering_trigrams/neutros/distribucion_docs/distribucion_docs_k5.csv')
df_pos = pd.read_csv('resultados_clustering_trigrams/positivos/distribucion_docs/distribucion_docs_k6.csv')

# Renombramos las columnas de probabilidad en cada archivo ANTES de unirlos
df_neg.rename(columns=nombres_neg, inplace=True)
df_neu.rename(columns=nombres_neu, inplace=True)
df_pos.rename(columns=nombres_pos, inplace=True)

# Unimos los tres archivos en uno solo
df_master = pd.concat([df_neg, df_neu, df_pos], ignore_index=True)

# Rellenamos los huecos vacíos con 0 (ej: una reseña positiva tendrá 0% en las quejas negativas)
columnas_probabilidad = list(nombres_neg.values()) + list(nombres_neu.values()) + list(nombres_pos.values())
df_master[columnas_probabilidad] = df_master[columnas_probabilidad].fillna(0)

# ==========================================
# 2. LIMPIEZA GEOGRÁFICA
# ==========================================
df_master[['City', 'Country']] = df_master['location'].str.split(', ', n=1, expand=True)
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
# 4. MAPEO DE TÓPICOS (Ganador Absoluto / Hard Clustering)
# ==========================================
# Mantenemos esto para los gráficos simples de burbujas y mapas
def nombrar_clusters(row):
    if row['Grupo_Sentimiento'] == 'Negativo':
        nombres_hard_neg = {0: "Pagos y dinero", 1: "Perfiles falsos y bots", 2: "Errores y verificación",
                            3: "Falta de matches", 4: "Quejas generales", 5: "Cuentas bloqueadas"}
        return nombres_hard_neg.get(row['Cluster_asignado'], "Otros negativos")

    elif row['Grupo_Sentimiento'] == 'Neutro':
        nombres_hard_neu = {0: "Ver quién te dio like", 1: "Fallos en cuentas de pago", 2: "Cansancio de la app",
                            3: "Mejoras de diseño", 4: "Límite de mensajes gratis"}
        return nombres_hard_neu.get(row['Cluster_asignado'], "Otros neutros")

    elif row['Grupo_Sentimiento'] == 'Positivo':
        nombres_hard_pos = {0: "Útil para viajar", 1: "Cara pero funciona", 2: "Fácil de usar", 3: "Para hacer amigos",
                            4: "Casos de éxito / Ayuda", 5: "Citas rápidas"}
        return nombres_hard_pos.get(row['Cluster_asignado'], "Otros positivos")


df_master['Nombre_Topico'] = df_master.apply(nombrar_clusters, axis=1)

# ==========================================
# 4.5. SENTIMIENTO DOMINANTE POR CIUDAD Y PAÍS
# ==========================================
df_master['Sentimiento_Dominante_Ciudad'] = df_master.groupby(['City', 'Country'])['Grupo_Sentimiento'].transform(
    lambda x: x.value_counts().idxmax() if not x.empty else 'Desconocido')

# ==========================================
# 5. EXPORTAR EL ARCHIVO FINAL
# ==========================================
df_master.to_csv('Tinder_Corpus_Tableau.csv', index=False, encoding='utf-8')
print("¡Archivo maestro creado con éxito!")