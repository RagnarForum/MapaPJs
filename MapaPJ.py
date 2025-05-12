import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import re # Import regular expressions

# --- Page Configuration ---
st.set_page_config(
    page_title="Mapa de Técnicos Terceirizados",
    page_icon=" Mago::globe_with_meridians:",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data # Cache the data loading and processing
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            st.error(f"Erro ao ler o CSV: {e}")
            return pd.DataFrame() # Return empty dataframe on error

    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()

    # Fill NaN values in expertise columns with empty strings for easier processing
    expertise_cols = ['Mecânica', 'Elétrica', 'Eletrônica', 'Processo']
    for col in expertise_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            st.warning(f"Coluna '{col}' não encontrada no arquivo CSV.")
            df[col] = '' # Add column if missing

    # Combine city and state for geocoding robustness
    df['EnderecoCompleto'] = df['Cidade'].fillna('') + ', ' + df['UF'].fillna('')
    df['EnderecoCompleto'] = df['EnderecoCompleto'].str.strip(', ') # Clean up if city or UF is missing

    return df

# --- Geocoding Function with Caching and Rate Limiting ---
# Initialize geolocator
geolocator = Nominatim(user_agent="technician_mapper_app_" + str(time.time())) # Unique user agent
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, error_wait_seconds=5.0) # Respect Nominatim usage policy

@st.cache_data # Cache geocoding results
def get_coordinates(address):
    if not address or pd.isna(address):
        return None, None
    try:
        location = geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            # Try geocoding just the state if full address fails (basic fallback)
            state = address.split(',')[-1].strip()
            if state != address: # Avoid infinite loop if only state was provided
                location_state = geocode(state)
                if location_state:
                     # Return state coords but log warning
                     # st.sidebar.warning(f"Não foi possível localizar '{address}'. Usando coordenadas do estado '{state}'.")
                     return location_state.latitude, location_state.longitude
            return None, None # Failed
    except Exception as e:
        # st.sidebar.error(f"Erro de geocodificação para '{address}': {e}")
        return None, None

# --- Helper Functions for Parsing Expertise ---
def parse_expertise(text):
    """Parses the expertise string into categories and machines."""
    if not isinstance(text, str) or not text.strip():
        return {}, [] # Return empty dict and list if input is not valid string

    categories = {}
    all_machines_in_cell = []
    current_category = None
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('»'):
            current_category = line[1:].strip()
            categories[current_category] = []
        elif current_category:
            # Split machines by comma, strip whitespace
            machines = [m.strip() for m in line.split(',') if m.strip()]
            categories[current_category].extend(machines)
            all_machines_in_cell.extend(machines)

    return categories, list(set(all_machines_in_cell)) # Return unique machines for the cell

@st.cache_data
def get_all_unique_items(df, expertise_cols):
    """Gets all unique categories and machines from the expertise columns."""
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

# Load data
file_path = './[AT]TécnicosTerceirosMapa.csv'
df_original = load_data(file_path)

if df_original.empty:
    st.stop() # Stop execution if data loading failed

# --- Sidebar Filters ---
st.sidebar.header("Filtros")

# Get unique values for filters
states = sorted(df_original['UF'].dropna().unique())
all_expertise_areas = ['Mecânica', 'Elétrica', 'Eletrônica', 'Processo']
all_categories, all_machines = get_all_unique_items(df_original, all_expertise_areas)

# State filter
selected_states = st.sidebar.multiselect("Estado (UF):", options=states, default=[])

# City filter (dynamic based on state)
if selected_states:
    available_cities = sorted(df_original[df_original['UF'].isin(selected_states)]['Cidade'].dropna().unique())
else:
    available_cities = sorted(df_original['Cidade'].dropna().unique())
selected_cities = st.sidebar.multiselect("Cidade:", options=available_cities, default=[])

# Technician Name filter
technician_names = sorted(df_original['Nome'].dropna().unique())
selected_name = st.sidebar.selectbox("Nome do Técnico:", options=["Todos"] + technician_names, index=0)

# Expertise Area filter
selected_area = st.sidebar.selectbox("Área de Atuação:", options=["Todas"] + all_expertise_areas, index=0)

# Machine Category filter
selected_category = st.sidebar.selectbox("Categoria de Máquina:", options=["Todas"] + all_categories, index=0)

# Specific Machine filter
selected_machine = st.sidebar.selectbox("Máquina Específica:", options=["Todas"] + all_machines, index=0)

# --- Filtering Data ---
df_filtered = df_original.copy()

if selected_states:
    df_filtered = df_filtered[df_filtered['UF'].isin(selected_states)]
if selected_cities:
    df_filtered = df_filtered[df_filtered['Cidade'].isin(selected_cities)]
if selected_name != "Todos":
    df_filtered = df_filtered[df_filtered['Nome'] == selected_name]

# Function to check if expertise matches filters
def check_expertise(row, area, category, machine):
    expertise_cols_to_check = all_expertise_areas if area == "Todas" else [area]
    match_found = False
    full_text_for_tooltip = []

    for col_name in expertise_cols_to_check:
        if col_name in row and pd.notna(row[col_name]) and row[col_name].strip():
            text = row[col_name]
            full_text_for_tooltip.append(f"**{col_name}:**\n{text}") # Add formatted text for tooltip

            # If only area is selected, any content in that area is a match
            if area != "Todas" and category == "Todas" and machine == "Todas":
                 match_found = True
                 continue # Check next area if needed (though loop is constrained by selected_area)

            # Check category and machine within this area's text
            cat_match = (category == "Todas") or (f"»{category}" in text)
            # Use regex for machine search to handle variations and word boundaries potentially
            machine_match = (machine == "Todas") or bool(re.search(r'(?<!\w)' + re.escape(machine) + r'(?!\w)', text, re.IGNORECASE))

            if cat_match and machine_match:
                 # If area filter is 'Todas', finding a match in *any* area is enough
                 if area == "Todas":
                    match_found = True
                    # Don't break here if area is 'Todas', need to collect all tooltips
                 # If a specific area is selected, only a match in *that* area counts
                 elif col_name == area:
                    match_found = True
                    break # Found match in the specific area, no need to check others

    # If the area filter was 'Todas' and no specific cat/machine was selected,
    # the simple existence check (area column not empty) should suffice.
    if area == "Todas" and category == "Todas" and machine == "Todas":
        match_found = any(pd.notna(row[col]) and row[col].strip() for col in expertise_cols_to_check if col in row)


    tooltip_content = "\n\n".join(full_text_for_tooltip) if full_text_for_tooltip else "Nenhuma especialidade detalhada."
    return pd.Series([match_found, tooltip_content])


# Apply expertise filters only if they are selected (not 'Todas')
if selected_area != "Todas" or selected_category != "Todas" or selected_machine != "Todas":
     # Apply the check_expertise function row-wise
     expertise_check_results = df_filtered.apply(
         lambda row: check_expertise(row, selected_area, selected_category, selected_machine),
         axis=1
     )
     # Assign the results back to the DataFrame (temporarily)
     df_filtered[['match', 'tooltip_info']] = expertise_check_results
     # Keep only rows that matched
     df_filtered = df_filtered[df_filtered['match'] == True]
else:
    # If no expertise filter is applied, generate tooltip info for all rows in df_filtered
    df_filtered['tooltip_info'] = df_filtered.apply(
        lambda row: "\n\n".join([f"**{col}:**\n{row[col]}" for col in all_expertise_areas if col in row and pd.notna(row[col]) and row[col].strip()] or ["Nenhuma especialidade detalhada."]),
        axis=1
    )


# --- Geocode Filtered Data ---
# Check if coordinates already exist to avoid re-running
if 'Latitude' not in df_filtered.columns or 'Longitude' not in df_filtered.columns:
    df_filtered[['Latitude', 'Longitude']] = None # Initialize columns if they don't exist

# Apply geocoding only to rows that don't have valid coordinates yet and have an address
rows_to_geocode = df_filtered[
    (df_filtered['Latitude'].isna() | df_filtered['Longitude'].isna()) &
    df_filtered['EnderecoCompleto'].notna() &
    (df_filtered['EnderecoCompleto'] != '')
].index

if not rows_to_geocode.empty:
    st.info(f"Geocodificando {len(rows_to_geocode)} endereços... Isso pode levar um tempo.")
    progress_bar = st.progress(0)
    total_rows = len(rows_to_geocode)

    coordinates = {}
    for i, idx in enumerate(rows_to_geocode):
        address = df_filtered.loc[idx, 'EnderecoCompleto']
        lat, lon = get_coordinates(address)
        coordinates[idx] = (lat, lon)
        # Update progress bar
        progress_bar.progress((i + 1) / total_rows)

    # Update DataFrame efficiently
    lat_updates = {idx: coord[0] for idx, coord in coordinates.items()}
    lon_updates = {idx: coord[1] for idx, coord in coordinates.items()}
    df_filtered['Latitude'].update(pd.Series(lat_updates))
    df_filtered['Longitude'].update(pd.Series(lon_updates))

    progress_bar.empty() # Remove progress bar after completion


# --- Display Map ---
st.subheader("Mapa de Técnicos")

# Create map centered roughly on Brazil or based on filtered data
map_center = [-14.2350, -51.9253] # Center of Brazil
zoom_start = 4

# Filter out rows without valid coordinates for mapping
df_map = df_filtered.dropna(subset=['Latitude', 'Longitude'])

if not df_map.empty:
    # Adjust map center and zoom based on data points
    map_center = [df_map['Latitude'].mean(), df_map['Longitude'].mean()]
    # Adjust zoom based on the spread of points (simple approach)
    lat_range = df_map['Latitude'].max() - df_map['Latitude'].min()
    lon_range = df_map['Longitude'].max() - df_map['Longitude'].min()
    if lat_range < 2 and lon_range < 2 :
        zoom_start = 10
    elif lat_range < 10 and lon_range < 10:
        zoom_start = 6
    else:
        zoom_start = 4


m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")

# Add markers to the map
for idx, row in df_map.iterrows():
    # Prepare popup content (HTML for basic formatting)
    popup_html = f"""
    <b>Nome:</b> {row['Nome']}<br>
    <b>Empresa:</b> {row['Empresa']}<br>
    <b>Local:</b> {row['Cidade']}, {row['UF']}<br>
    <hr>
    <b>Especialidades Filtradas/Todas:</b><br>
    <pre style="white-space: pre-wrap; word-wrap: break-word; max-height: 200px; overflow-y: auto;">{row.get('tooltip_info', 'N/A').replace('**','<b>').replace('**','</b>').replace('\\n','<br>')}</pre>
    """
    # Use get to safely access tooltip_info, provide default if missing

    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_html, max_width=350), # Adjust max_width as needed
        tooltip=f"{row['Nome']} ({row['Cidade']})" # Tooltip on hover
    ).add_to(m)

# Display the map
st_folium(m, width='100%', height=500) # Use st_folium

if df_map.empty and not df_filtered.empty:
    st.warning("Não foi possível geolocalizar os técnicos filtrados. Verifique os endereços no CSV.")
elif df_filtered.empty:
    st.info("Nenhum técnico encontrado com os filtros selecionados.")


# --- Display Filtered Data Table ---
st.subheader("Dados dos Técnicos Filtrados")
st.write(f"Total de técnicos encontrados: {len(df_filtered)}")

# Select and reorder columns for display
columns_to_display = ['Nome', 'Empresa', 'Cidade', 'UF', 'Mecânica', 'Elétrica', 'Eletrônica', 'Processo']
# Ensure only existing columns are selected
columns_to_display = [col for col in columns_to_display if col in df_filtered.columns]

st.dataframe(df_filtered[columns_to_display], use_container_width=True)

# Optional: Show raw filtered data for debugging or detail
with st.expander("Ver dados brutos filtrados (inclui coordenadas)"):
    st.dataframe(df_filtered)

# --- Footer/Info ---
st.sidebar.markdown("---")
st.sidebar.info("Aplicação desenvolvida para visualização de técnicos.")
