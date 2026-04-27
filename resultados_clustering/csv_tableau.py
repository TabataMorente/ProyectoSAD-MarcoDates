import pandas as pd

# ==========================================
# 1. CONSOLIDACIÓN DE CSVS
# ==========================================
# Carga tus tres archivos de clústeres
df_neg = pd.read_csv('./negativos/clusters_k5.csv')
df_neu = pd.read_csv('./neutros/clusters_k8.csv')
df_pos = pd.read_csv('./positivos/clusters_k6.csv')

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
# 4. MAPEO DE TÓPICOS (Nombres Formales)
# ==========================================
def nombrar_clusters(row):
    # Si es Negativo (k=5)
    if row['Grupo_Sentimiento'] == 'Negativo':
        nombres_neg = {
            0: "Problemas de Pagos y Suscripciones",
            1: "Falta de Matches y Resultados",
            2: "Errores Técnicos y de Acceso",
            3: "Perfiles Falsos y Bots",
            4: "Experiencia General y Spam"
        }
        return nombres_neg.get(row['Cluster_LDA'], "Otro Negativo")

    # Si es Neutro (k=8) -> ¡Nuevos nombres corporativos!
    elif row['Grupo_Sentimiento'] == 'Neutro':
        nombres_neu = {
            0: "Opciones Específicas de Perfil",
            1: "Fallos de Verificación y Login",
            2: "Filtro de Distancia Forzado",
            3: "Pago por ver 'Me gusta'",
            4: "Diseño UX y Usabilidad",
            5: "Errores de Perfil y Premium",
            6: "Notificaciones Fantasma",
            7: "Baneos y Pagos Locales"
        }
        return nombres_neu.get(row['Cluster_LDA'], "Otro Neutro")

    # Si es Positivo (k=6)
    elif row['Grupo_Sentimiento'] == 'Positivo':
        nombres_pos = {
            0: "Facilidad para Conocer Gente",
            1: "Interfaz Intuitiva y Fluida",
            2: "Alta Calidad de Perfiles",
            3: "Resultados Rápidos",
            4: "Experiencia Positiva General",
            5: "Comparativa Favorable con Otras Apps"
        }
        return nombres_pos.get(row['Cluster_LDA'], "Otro Positivo")

# Aplicamos la función para crear la columna final de tópicos
df_master['Nombre_Topico'] = df_master.apply(nombrar_clusters, axis=1)

# ==========================================
# 5. EXPORTAR EL ARCHIVO FINAL
# ==========================================
df_master.to_csv('Tinder_Corpus_Tableau.csv', index=False, encoding='utf-8')
print("¡Archivo maestro creado con éxito!")