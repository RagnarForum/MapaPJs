import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import re
import os # <<< ADICIONE ESTA LINHA NO INÍCIO DO SEU SCRIPT
from folium import plugins

# --- Page Configuration ---
st.set_page_config(
    page_title="Mago: Mapa de Técnicos",
    page_icon=" Mago::globe_with_meridians:",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path_internal): # Renomeado para evitar conflito de nome
    try:
        df = pd.read_csv(file_path_internal, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path_internal, encoding='latin1')
        except Exception as e:
            st.error(f"Erro ao ler o CSV '{file_path_internal}': {e}")
            return pd.DataFrame()
    except FileNotFoundError: # Captura específica para FileNotFoundError dentro da função
        st.error(f"DENTRO DE LOAD_DATA: Arquivo '{file_path_internal}' não encontrado.")
        return pd.DataFrame()


    df.columns = df.columns.str.strip()
    expertise_cols = ['Mecânica', 'Elétrica', 'Eletrônica', 'Processo']
    for col in expertise_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            st.warning(f"Coluna '{col}' não encontrada no arquivo CSV.")
            df[col] = ''

    df['EnderecoCompleto'] = df['Cidade'].fillna('') + ', ' + df['UF'].fillna('')
    df['EnderecoCompleto'] = df['EnderecoCompleto'].str.strip(', ')
    return df

# --- Geocoding Function with Caching and Rate Limiting ---
# ... (resto das funções de geocodificação e parse_expertise permanecem iguais) ...
geolocator = Nominatim(user_agent="technician_mapper_app_" + str(time.time()) + "_v3") # Alterei o user_agent para limpar cache se necessário
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, error_wait_seconds=10.0, max_retries=2)

@st.cache_data
def get_coordinates(address):
    if not address or pd.isna(address):
        return None, None
    try:
        location = geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            state_search = address.split(',')[-1].strip()
            if state_search and state_search != address:
                location_state = geocode(state_search, timeout=10)
                if location_state:
                    return location_state.latitude, location_state.longitude
            return None, None
    except Exception as e:
        return None, None

def parse_expertise(text):
    if not isinstance(text, str) or not text.strip():
        return {}, []
    categories = {}
    all_machines_in_cell = []
    current_category = None
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith('»'):
            current_category = line[1:].strip()
            categories[current_category] = []
        elif current_category:
            machines = [m.strip() for m in line.split(',') if m.strip()]
            categories[current_category].extend(machines)
            all_machines_in_cell.extend(machines)
    return categories, list(set(all_machines_in_cell))

@st.cache_data
def get_all_unique_items(df, expertise_cols):
    all_categories = set()
    all_machines = set()
    for col in expertise_cols:
        if col in df.columns:
            for text in df[col].dropna():
                parsed_cats, parsed_machines = parse_expertise(text)
                all_categories.update(parsed_cats.keys())
                all_machines.update(parsed_machines)
    return sorted(list(all_categories)), sorted(list(all_machines))


# --- Main App Logic ---
st.title(" Mago: Mapeamento e Consulta de Técnicos Parceiros")
st.markdown("Utilize os filtros na barra lateral para encontrar técnicos por localização ou especialidade.")

# --- DEFINIÇÃO DO CAMINHO DO ARQUIVO ---
# Certifique-se de que este nome é EXATAMENTE igual ao nome do seu arquivo CSV
# incluindo maiúsculas/minúsculas, espaços e acentos.
FILE_NAME = '[AT]TécnicosTerceirosMapa.csv'

# Constrói o caminho absoluto para o arquivo, assumindo que ele está no mesmo diretório do script
# Isso é mais robusto, especialmente para deploy.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, FILE_NAME)

# --- Verificação da Existência do Arquivo ---
if not os.path.exists(file_path):
    st.error(f"ERRO CRÍTICO: O arquivo de dados '{FILE_NAME}' não foi encontrado.")
    st.error(f"O script está procurando por ele em: '{file_path}'")
    st.error(f"Por favor, verifique se:")
    st.error(f"1. O arquivo '{FILE_NAME}' está no MESMO DIRETÓRIO que o seu script Python (`app.py`).")
    st.error(f"2. O nome do arquivo no script está EXATAMENTE IGUAL ao nome do seu arquivo (cuidado com letras maiúsculas/minúsculas, espaços e acentos como 'é', 'ã', 'ç').")
    st.info(f"Conteúdo do diretório atual ({BASE_DIR}):")
    try:
        st.write(os.listdir(BASE_DIR))
    except Exception as e:
        st.write(f"Não foi possível listar o conteúdo do diretório: {e}")
    st.stop() # Para a execução do script se o arquivo não for encontrado

df_original = load_data(file_path) # Agora passamos o file_path absoluto

if df_original.empty:
    st.warning("O DataFrame está vazio após tentar carregar os dados. Verifique se o arquivo CSV tem conteúdo ou se houve erros durante o carregamento.")
    st.stop()

# --- Sidebar Filters ---
# ... (o resto do seu código para filtros, geocodificação de dados filtrados, mapa, etc., continua aqui)
# Certifique-se de que nenhuma outra parte do código está redefinindo 'file_path' para um valor relativo problemático.

st.sidebar.header("Filtros")
states = sorted(df_original['UF'].dropna().unique())
all_expertise_areas = ['Mecânica', 'Elétrica', 'Eletrônica', 'Processo']
all_categories, all_machines = get_all_unique_items(df_original, all_expertise_areas)

selected_states = st.sidebar.multiselect("Estado (UF):", options=states, default=[])
if selected_states:
    available_cities = sorted(df_original[df_original['UF'].isin(selected_states)]['Cidade'].dropna().unique())
else:
    available_cities = sorted(df_original['Cidade'].dropna().unique())
selected_cities = st.sidebar.multiselect("Cidade:", options=available_cities, default=[])
technician_names = sorted(df_original['Nome'].dropna().unique())
selected_name = st.sidebar.selectbox("Nome do Técnico:", options=["Todos"] + technician_names, index=0)
selected_area = st.sidebar.selectbox("Área de Atuação:", options=["Todas"] + all_expertise_areas, index=0)
selected_category = st.sidebar.selectbox("Categoria de Máquina:", options=["Todas"] + all_categories, index=0)
selected_machine = st.sidebar.selectbox("Máquina Específica:", options=["Todas"] + all_machines, index=0)

# --- Filtering Data ---
df_filtered = df_original.copy()
if selected_states:
    df_filtered = df_filtered[df_filtered['UF'].isin(selected_states)]
if selected_cities:
    df_filtered = df_filtered[df_filtered['Cidade'].isin(selected_cities)]
if selected_name != "Todos":
    df_filtered = df_filtered[df_filtered['Nome'] == selected_name]

def check_expertise(row, area, category, machine):
    expertise_cols_to_check = all_expertise_areas if area == "Todas" else [area]
    match_found = False
    full_text_for_tooltip = []

    for col_name in expertise_cols_to_check:
        if col_name in row and pd.notna(row[col_name]) and row[col_name].strip():
            text = str(row[col_name])
            full_text_for_tooltip.append(f"**{col_name}:**\n{text}")

            if area != "Todas" and category == "Todas" and machine == "Todas":
                match_found = True
                continue

            cat_match = (category == "Todas") or (f"»{category}" in text)
            machine_match = (machine == "Todas") or bool(re.search(r'(?<!\w)' + re.escape(machine) + r'(?!\w)', text, re.IGNORECASE))

            if cat_match and machine_match:
                if area == "Todas":
                    match_found = True
                elif col_name == area:
                    match_found = True
                    break
    
    if area == "Todas" and category == "Todas" and machine == "Todas":
        match_found = any(pd.notna(row[col]) and str(row[col]).strip() for col in expertise_cols_to_check if col in row)

    tooltip_content = "\n\n".join(full_text_for_tooltip) if full_text_for_tooltip else "Nenhuma especialidade detalhada."
    return pd.Series([match_found, tooltip_content])

if selected_area != "Todas" or selected_category != "Todas" or selected_machine != "Todas":
    expertise_check_results = df_filtered.apply(
        lambda row: check_expertise(row, selected_area, selected_category, selected_machine),
        axis=1
    )
    df_filtered[['match', 'tooltip_info']] = expertise_check_results
    df_filtered = df_filtered[df_filtered['match'] == True].drop(columns=['match'])
else:
    df_filtered['tooltip_info'] = df_filtered.apply(
        lambda row: "\n\n".join([f"**{col}:**\n{str(row[col])}" for col in all_expertise_areas if col in row and pd.notna(row[col]) and str(row[col]).strip()] or ["Nenhuma especialidade detalhada."]),
        axis=1
    )

# --- Geocode Filtered Data ---
if 'Latitude' not in df_filtered.columns or 'Longitude' not in df_filtered.columns:
    df_filtered['Latitude'] = pd.NA
    df_filtered['Longitude'] = pd.NA

df_filtered['Latitude'] = pd.to_numeric(df_filtered['Latitude'], errors='coerce')
df_filtered['Longitude'] = pd.to_numeric(df_filtered['Longitude'], errors='coerce')

rows_to_geocode_mask = (df_filtered['Latitude'].isna() | df_filtered['Longitude'].isna()) & \
                       df_filtered['EnderecoCompleto'].notna() & \
                       (df_filtered['EnderecoCompleto'] != '')
rows_to_geocode_indices = df_filtered[rows_to_geocode_mask].index

if not rows_to_geocode_indices.empty:
    st.info(f"Geocodificando {len(rows_to_geocode_indices)} novos endereços... Isso pode levar um tempo.")
    progress_bar = st.progress(0)
    
    coordinates_lat = {}
    coordinates_lon = {}

    for i, idx in enumerate(rows_to_geocode_indices):
        address = df_filtered.loc[idx, 'EnderecoCompleto']
        lat, lon = get_coordinates(address)
        if lat is not None and lon is not None:
            coordinates_lat[idx] = lat
            coordinates_lon[idx] = lon
        else:
            coordinates_lat[idx] = pd.NA
            coordinates_lon[idx] = pd.NA
        progress_bar.progress((i + 1) / len(rows_to_geocode_indices))
    
    df_filtered.loc[coordinates_lat.keys(), 'Latitude'] = pd.Series(coordinates_lat)
    df_filtered.loc[coordinates_lon.keys(), 'Longitude'] = pd.Series(coordinates_lon)
    
    # Atualizar df_original também para persistir geocodificações na sessão
    # Isso é uma simplificação. Para persistência real entre execuções, seria necessário salvar o df_original.
    df_original.update(df_filtered[['Latitude', 'Longitude']])

    progress_bar.empty()

# --- Display Map ---
st.subheader("Mapa de Técnicos")
brazil_bounds = [[-33.75, -73.99], [5.28, -34.70]]
map_center_br = [-14.2350, -51.9253]
zoom_start_br = 4

df_map = df_filtered.dropna(subset=['Latitude', 'Longitude'])
current_map_center = map_center_br
current_zoom = zoom_start_br

if not df_map.empty:
    current_map_center = [df_map['Latitude'].mean(), df_map['Longitude'].mean()]
    lat_std = df_map['Latitude'].std()
    lon_std = df_map['Longitude'].std()
    if pd.notna(lat_std) and pd.notna(lon_std) and lat_std < 0.5 and lon_std < 0.5 and len(df_map) == 1:
        current_zoom = 12
    elif pd.notna(lat_std) and pd.notna(lon_std) and lat_std < 2 and lon_std < 2:
        current_zoom = 7
    elif pd.notna(lat_std) and pd.notna(lon_std) and lat_std < 5 and lon_std < 5:
        current_zoom = 5
    else:
        current_zoom = zoom_start_br

selected_tile = "Esri_NatGeoWorldMap"
m = folium.Map(
    location=current_map_center,
    zoom_start=current_zoom,
    tiles=selected_tile,
    attr='Tiles © Esri — National Geographic, Esri, DeLorme, NAVTEQ, UNEP-WCMC, USGS, NASA, ESA, METI, NRCAN, GEBCO, NOAA, iPC' if selected_tile == "Esri_NatGeoWorldMap" else None,
    control_scale=True
)
minimap = plugins.MiniMap()
m.add_child(minimap)


for idx, row in df_map.iterrows():
    tooltip_text = f"{row['Nome']} ({row['Cidade']})"
    popup_html = f"""
    <div style="font-family: Arial, sans-serif; font-size: 13px;">
    <strong>Nome:</strong> {row['Nome']}<br>
    <strong>Empresa:</strong> {row['Empresa']}<br>
    <strong>Local:</strong> {row['Cidade']}, {row['UF']}<br>
    <hr style="margin: 5px 0;">
    <strong>Especialidades:</strong><br>
    <div style="max-height: 150px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; background-color: #f9f9f9; border: 1px solid #eee; padding: 5px; margin-top: 3px;">{row.get('tooltip_info', 'N/A').replace('**','<b>').replace('**','</b>').replace('\\n','<br>')}</div>
    </div>
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_html, max_width=350, min_width=300),
        tooltip=tooltip_text,
        icon=folium.Icon(color="blue", icon="wrench", prefix="fa")
    ).add_to(m)

map_data_returned = st_folium(m, width='100%', height=700, returned_objects=[])

if df_map.empty and not df_filtered.empty:
    st.warning("Não foi possível geolocalizar os técnicos filtrados. Verifique os endereços ou aguarde a geocodificação.")
elif df_filtered.empty:
    st.info("Nenhum técnico encontrado com os filtros selecionados.")

# --- Display Filtered Data Table ---
st.subheader("Dados dos Técnicos Filtrados")
st.write(f"Total de técnicos encontrados: {len(df_filtered)}")
columns_to_display = ['Nome', 'Empresa', 'Cidade', 'UF', 'Mecânica', 'Elétrica', 'Eletrônica', 'Processo']
columns_to_display = [col for col in columns_to_display if col in df_filtered.columns]
st.dataframe(df_filtered[columns_to_display], use_container_width=True, height=(min(len(df_filtered) + 1, 10) * 35) + 3)

with st.expander("Ver dados brutos filtrados (inclui coordenadas e detalhes de especialidade)"):
    st.dataframe(df_filtered)

# --- Footer/Info ---
st.sidebar.markdown("---")
st.sidebar.info("Aplicação Mago para visualização e consulta de técnicos.")
