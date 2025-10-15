
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from datetime import datetime
from io import StringIO

st.set_page_config(page_title="Small Dams (CSV Robust v3)", page_icon="🏞️", layout="wide")

# -------------------------
# Loading (robust encodings + sep auto-detect)
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    path = "All_Damsnew.csv"
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, sep=None, engine="python")
            break
        except Exception as e:
            last_err = e
            df = None
            continue
    if df is None:
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(raw), sep=None, engine="python")
    # Normalize headers and values
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()
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
    df[col_name] = df[col_name].replace({"": np.nan})
    df = df.dropna(subset=[col_lat, col_lon]).copy()
    df[col_lat] = pd.to_numeric(df[col_lat], errors="coerce").astype("float32")
    df[col_lon] = pd.to_numeric(df[col_lon], errors="coerce").astype("float32")

    if col_height: df[col_height] = coerce_num(df[col_height]).astype("float32")
    if col_cca:    df[col_cca]    = coerce_num(df[col_cca]).astype("float32")
    if col_year:
        years = coerce_num(df[col_year])
        df["Age (years)"] = (datetime.now().year - years).astype("Int16")
    else:
        df["Age (years)"] = pd.Series([np.nan]*len(df), dtype="float32")

    df[col_district] = df[col_district].astype("category")
    df[col_name]     = df[col_name].astype("category")
    return df

def build_rank_table(frame, col_name, col_district, metric_col, nice_label, top_n=10):
    t = frame[[col_name, col_district, metric_col]].dropna(subset=[metric_col]).copy()
    t[col_name] = t[col_name].astype(str).replace({"nan": "—", "": "—"})
    t = t.sort_values(metric_col, ascending=False, na_position="last").head(top_n).reset_index(drop=True)
    t.index = t.index + 1
    t.index.name = "Rank"
    t = t.rename(columns={metric_col: nice_label})
    return t

def comparison_text(row, frame, metric_col, nice_label, col_name, scope_label="Organization"):
    if metric_col is None or pd.isna(row.get(metric_col, np.nan)):
        return f"{nice_label}: not available for this dam."
    value = float(row[metric_col])
    scope = frame.dropna(subset=[metric_col])
    if scope.empty:
        return f"{nice_label}: not available in {scope_label}."
    max_val = scope[metric_col].max()
    min_val = scope[metric_col].min()
    max_dam = scope.loc[scope[metric_col].idxmax(), col_name]
    min_dam = scope.loc[scope[metric_col].idxmin(), col_name]
    if np.isclose(value, max_val, equal_nan=False):
        return f"{nice_label}: This is the **highest** in the {scope_label} ({value:,.0f})."
    if np.isclose(value, min_val, equal_nan=False):
        return f"{nice_label}: This is the **lowest** in the {scope_label} ({value:,.0f})."
    diff = max_val - value
    return f"{nice_label}: {value:,.0f} — **{diff:,.0f} lower** than the highest in the {scope_label} ({max_val:,.0f} at {max_dam})."

# -------------------------
# Load & detect columns
# -------------------------
try:
    df, alias = load_data()
except Exception as e:
    st.error(f"Failed to load All_Dams.csv: {e}")
    st.stop()

col_district = find_col(alias, ["district"])
col_name     = find_col(alias, ["name of dam", "dam name"])
col_lat      = find_col(alias, ["latitude"])
col_lon      = find_col(alias, ["longitude"])
col_height   = find_col(alias, ["height"])
col_cca      = find_col(alias, ["c.c.a", "cca"])
col_year     = find_col(alias, ["year of completion", "year"])

required = [col_district, col_name, col_lat, col_lon]
missing = [c for c in required if c is None]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = prepare(df, col_district, col_name, col_height, col_cca, col_year, col_lat, col_lon)

# -------------------------
# UI
# -------------------------
st.title("🏞️ Small Dams Explorer")
st.caption("Bright basemap + highlighted selection.")

# Filters
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

# Selected summary
st.markdown("---")
c1, c2 = st.columns([1,2])

with c1:
    st.subheader("🎯 Selected")
    if dam != "All":
        row = scope_df[scope_df[col_name] == dam].iloc[0]
        st.markdown(f"**{row[col_name]}**")
        st.markdown("**Organization comparison**")
        st.write(comparison_text(row, df, col_height, "Height (ft)", col_name, "organization"))
        st.write(comparison_text(row, df, col_cca, "C.C.A (Acres)", col_name, "organization"))
        st.write(comparison_text(row, df, "Age (years)", "Age (years)", col_name, "organization"))
        if district != "All":
            st.markdown(f"**{district} comparison**")
            st.write(comparison_text(row, scope_df, col_height, "Height (ft)", col_name, f"{district}"))
            st.write(comparison_text(row, scope_df, col_cca, "C.C.A (Acres)", col_name, f"{district}"))
            st.write(comparison_text(row, scope_df, "Age (years)", "Age (years)", col_name, f"{district}"))
    else:
        st.info("Pick a dam to see comparison statements.")

with c2:
    st.subheader("🗺️ Map")
    if not perf_mode:
        center_lat = float(scope_df[col_lat].median()) if len(scope_df) else 30.0
        center_lon = float(scope_df[col_lon].median()) if len(scope_df) else 70.0
        mdf = scope_df[[col_lon, col_lat, col_name, col_district]].copy()
        mdf[col_name] = mdf[col_name].astype(str).replace({"nan": "—", "": "—"})
        mdf["is_selected"] = (mdf[col_name] == dam) if dam != "All" else False

        # Bright basemap (Carto Positron, no API key)
        base = pdk.Layer(
            "TileLayer",
            data="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            subdomains=["a","b","c","d"],
        )

        # All dams (neutral dark dots)
        layer_all = pdk.Layer(
            "ScatterplotLayer",
            data=mdf[~mdf["is_selected"]],
            get_position=[col_lon, col_lat],
            get_radius=2000,
            get_fill_color=[30, 30, 30, 160],
            get_line_color=[255,255,255,120],
            line_width_min_pixels=0.5,
            pickable=True,
        )

        # Selected dam (bright yellow with white stroke)
        layer_sel = pdk.Layer(
            "ScatterplotLayer",
            data=mdf[mdf["is_selected"]],
            get_position=[col_lon, col_lat],
            get_radius=4000,
            get_fill_color=[255, 215, 0, 220],
            get_line_color=[255,255,255,255],
            line_width_min_pixels=2,
            pickable=True,
        )

        tooltip = {"html": f"<b>{col_name}</b>: {{{{{col_name}}}}}<br/><b>District</b>: {{{{{col_district}}}}}"}

        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=8),
            layers=[base, layer_all, layer_sel],
            tooltip=tooltip,
            map_provider=None,  # use the TileLayer, not Mapbox
        )
        st.pydeck_chart(deck, use_container_width=True)
    else:
        st.info("Performance mode is ON — map skipped.")

# Rankings
st.markdown("---")
tabs = st.tabs(["Organization-wide"] + ([f"{district} only"] if district != "All" else []))

with tabs[0]:
    st.markdown("### Top by Height (ft)")
    st.dataframe(build_rank_table(df, col_name, col_district, col_height, "Height (ft)"), use_container_width=True)
    st.markdown("### Top by C.C.A (Acres)")
    st.dataframe(build_rank_table(df, col_name, col_district, col_cca, "C.C.A (Acres)"), use_container_width=True)
    st.markdown("### Top by Age (years)")
    st.dataframe(build_rank_table(df, col_name, col_district, "Age (years)", "Age (years)"), use_container_width=True)

if district != "All":
    with tabs[1]:
        st.markdown(f"### Top by Height (ft) — {district}")
        st.dataframe(build_rank_table(scope_df, col_name, col_district, col_height, "Height (ft)"), use_container_width=True)
        st.markdown(f"### Top by C.C.A (Acres) — {district}")
        st.dataframe(build_rank_table(scope_df, col_name, col_district, col_cca, "C.C.A (Acres)"), use_container_width=True)
        st.markdown(f"### Top by Age (years) — {district}")
        st.dataframe(build_rank_table(scope_df, col_name, col_district, "Age (years)", "Age (years)"), use_container_width=True)
