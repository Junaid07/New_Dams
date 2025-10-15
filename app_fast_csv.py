
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from datetime import datetime

st.set_page_config(page_title="Small Dams (CSV, Fast)", page_icon="⚡", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    # Hardcoded CSV file
    path = "All_Dams.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    alias = {c.lower().replace("\n"," "): c for c in df.columns}
    return df, alias

def find_col(alias, keys):
    for k in alias.keys():
        for t in keys:
            if t in k:
                return alias[k]
    return None

def coerce_num(series):
    return pd.to_numeric(series, errors="coerce")

@st.cache_data(show_spinner=False)
def prepare(df, col_district, col_name, col_height, col_cca, col_year, col_lat, col_lon):
    df = df.dropna(subset=[col_lat, col_lon]).copy()
    df[col_lat] = pd.to_numeric(df[col_lat], errors="coerce").astype("float32")
    df[col_lon] = pd.to_numeric(df[col_lon], errors="coerce").astype("float32")

    if col_height: df[col_height] = coerce_num(df[col_height]).astype("float32")
    if col_cca:    df[col_cca]    = coerce_num(df[col_cca]).astype("float32")
    if col_year:
        years = coerce_num(df[col_year]).astype("Int32")
        age = (datetime.now().year - years).astype("Int16")
        df["Age (years)"] = age
    else:
        df["Age (years)"] = np.int16(-1)

    df[col_district] = df[col_district].astype("category")
    df[col_name]     = df[col_name].astype("category")

    org_ranks = {}
    if col_height: org_ranks["Height (ft)"] = df[[col_name, col_district, col_height]].nlargest(10, col_height)
    if col_cca:    org_ranks["C.C.A (Acres)"] = df[[col_name, col_district, col_cca]].nlargest(10, col_cca)
    if "Age (years)" in df.columns:
        org_ranks["Age (years)"] = df[[col_name, col_district, "Age (years)"]].nlargest(10, "Age (years)")

    return df, org_ranks

@st.cache_data(show_spinner=False)
def district_ranks(df, district, col_district, col_name, col_height, col_cca):
    sub = df[df[col_district].astype(str) == district]
    out = {}
    if col_height: out["Height (ft)"] = sub[[col_name, col_district, col_height]].nlargest(10, col_height)
    if col_cca:    out["C.C.A (Acres)"] = sub[[col_name, col_district, col_cca]].nlargest(10, col_cca)
    if "Age (years)" in sub.columns:
        out["Age (years)"] = sub[[col_name, col_district, "Age (years)"]].nlargest(10, "Age (years)")
    return out

# Load
try:
    df, alias = load_data()
except FileNotFoundError:
    st.error("Could not find 'All_Dams.csv' next to the app.")
    st.stop()

# Columns
col_district = find_col(alias, ["district"])
col_name     = find_col(alias, ["name of dam", "dam name"])
col_lat      = find_col(alias, ["latitude"])
col_lon      = find_col(alias, ["longitude"])
col_height   = find_col(alias, ["height"])
col_cca      = find_col(alias, ["c.c.a", "cca"])
col_year     = find_col(alias, ["year of completion", "year"])
col_type     = find_col(alias, ["type of dam"])
col_status   = find_col(alias, ["operational", "status"])
col_river    = find_col(alias, ["river", "nullah"])

required = [col_district, col_name, col_lat, col_lon]
missing = [c for c in required if c is None]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df, org_ranks = prepare(df, col_district, col_name, col_height, col_cca, col_year, col_lat, col_lon)

# UI
st.title("⚡ Small Dams Explorer (CSV, Fast)")
st.caption("CSV-only build (no openpyxl).")

cols = st.columns([1,1,1])
districts = ["All"] + sorted([str(x) for x in df[col_district].dropna().unique().tolist()])
with cols[0]:
    district = st.selectbox("District", districts, index=0)
scope_df = df if district == "All" else df[df[col_district].astype(str) == district]
dams = ["All"] + sorted([str(x) for x in scope_df[col_name].dropna().unique().tolist()])
with cols[1]:
    dam = st.selectbox("Dam", dams, index=0)
with cols[2]:
    perf_mode = st.toggle("Performance mode (skip map)", value=False)

st.markdown("---")
c1, c2 = st.columns([1,2])

with c1:
    st.subheader("🎯 Selected")
    if dam != "All":
        row = scope_df[scope_df[col_name] == dam].iloc[0]
        st.markdown(f"**{row[col_name]}**")
        if col_type: st.write(f"Type: {row.get(col_type, '—')}")
        if col_status: st.write(f"Status: {row.get(col_status, '—')}")
        if col_river: st.write(f"River/Nullah: {row.get(col_river, '—')}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Height (ft)", "—" if pd.isna(row.get(col_height)) else f"{row.get(col_height):,.0f}")
        m2.metric("C.C.A (Acres)", "—" if pd.isna(row.get(col_cca)) else f"{row.get(col_cca):,.0f}")
        m3.metric("Age (years)", "—" if pd.isna(row.get('Age (years)')) else f"{row.get('Age (years)'):,.0f}")
    else:
        st.info("Pick a dam to see details.")

with c2:
    st.subheader("🗺️ Map")
    if not perf_mode:
        center_lat = float(scope_df[col_lat].median()) if len(scope_df) else 30.0
        center_lon = float(scope_df[col_lon].median()) if len(scope_df) else 70.0
        mdf = scope_df[[col_lon, col_lat, col_name, col_district]].copy()
        if dam != "All":
            mdf["size"] = np.where(mdf[col_name] == dam, 6000, 2500).astype("int32")
        else:
            mdf["size"] = 2500
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=mdf,
            get_position=[col_lon, col_lat],
            get_radius="size",
            pickable=True,
            opacity=0.7,
        )
        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=7),
            layers=[layer],
            tooltip={"text": "{%s} ({%s})" % (col_name, col_district)}
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("Performance mode is ON — map skipped.")

st.markdown("---")
st.subheader("📊 Rankings & Comparisons")
tabs = st.tabs(["Organization-wide"] + ([f"{district} only"] if district != "All" else []))

with tabs[0]:
    for label, tdf in org_ranks.items():
        st.markdown(f"**Top by {label}**")
        st.dataframe(tdf, use_container_width=True)

if district != "All":
    with tabs[1]:
        dr = district_ranks(df, district, col_district, col_name, col_height, col_cca)
        for label, tdf in dr.items():
            st.markdown(f"**Top by {label} — {district}**")
            st.dataframe(tdf, use_container_width=True)

if dam != "All":
    st.markdown("---")
    st.subheader("📈 Key attributes")
    row = scope_df[scope_df[col_name] == dam].iloc[0]
    data = []
    if col_height and pd.notna(row[col_height]): data.append(("Height (ft)", float(row[col_height])))
    if col_cca and pd.notna(row[col_cca]):       data.append(("C.C.A (Acres)", float(row[col_cca])))
    if "Age (years)" in scope_df.columns and pd.notna(row["Age (years)"]):
        data.append(("Age (years)", float(row["Age (years)"])))
    if data:
        kdf = pd.DataFrame(data, columns=["Feature", "Value"])
        st.bar_chart(kdf.set_index("Feature"))
    else:
        st.info("No numeric features available for this dam.")
