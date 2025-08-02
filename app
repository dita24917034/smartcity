import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Smart Traffic Dashboard", layout="wide")
st.title("🚦 Smart City Traffic Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("trafficMetaData.csv")
    df = df.dropna(subset=['DURATION_IN_SEC', 'DISTANCE_IN_METERS', 'POINT_1_LAT', 'POINT_1_LNG', 'NDT_IN_KMH'])
    return df

df = load_data()

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("🔎 Filter Data")

city_options = df['POINT_1_CITY'].dropna().unique()
selected_city = st.sidebar.selectbox("Pilih Kota Awal", options=city_options)

road_types = df['ROAD_TYPE'].dropna().unique()
selected_road = st.sidebar.selectbox("Pilih Jenis Jalan", options=road_types)

# Filter dataframe
filtered_df = df[(df['POINT_1_CITY'] == selected_city) & (df['ROAD_TYPE'] == selected_road)]

st.markdown(f"📍 **Menampilkan data untuk kota:** `{selected_city}` dan jenis jalan: `{selected_road}`")

# ----------------------------
# Prediksi Durasi Perjalanan
# ----------------------------
st.subheader("📈 Prediksi Durasi Perjalanan (Random Forest + Plotly)")

features = ['DISTANCE_IN_METERS', 'NDT_IN_KMH']
X = filtered_df[features]
y = filtered_df['DURATION_IN_SEC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plotly Line Chart
line_df = pd.DataFrame({
    'Sample': list(range(len(y_test))),
    'Actual Duration': y_test.values,
    'Predicted Duration': y_pred
})

fig1 = px.line(line_df, x='Sample', y=['Actual Duration', 'Predicted Duration'],
               title="Durasi Aktual vs Prediksi (Random Forest)", markers=True)
fig1.update_layout(legend_title_text='Legend', height=400)
st.plotly_chart(fig1, use_container_width=True)

# ----------------------------
# Klasterisasi Lokasi Jalan
# ----------------------------
st.subheader("📍 Klasterisasi Titik Jalan (KMeans + Plotly)")

location_data = filtered_df[['POINT_1_LAT', 'POINT_1_LNG']].dropna()
scaler = StandardScaler()
loc_scaled = scaler.fit_transform(location_data)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
labels = kmeans.fit_predict(loc_scaled)

cluster_df = location_data.copy()
cluster_df['Cluster'] = labels

fig2 = px.scatter_mapbox(
    cluster_df,
    lat='POINT_1_LAT',
    lon='POINT_1_LNG',
    color='Cluster',
    zoom=10,
    mapbox_style='carto-positron',
    title="Pemetaan Klaster Lokasi Titik Jalan"
)
fig2.update_layout(height=500)
st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# Evaluasi
# ----------------------------
mae = mean_absolute_error(y_test, y_pred)
st.metric("🎯 MAE (Mean Absolute Error)", f"{mae:.2f} detik")

