# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="CSV Explorer", page_icon="📊", layout="wide")

st.title("📊 CSView — Enhanced CSV Viewer")

# ----------------------
# File upload + read
# ----------------------
uploaded = st.file_uploader("Upload your CSV", type=["csv"])
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    # Use low_memory=False to avoid dtype warnings for large CSVs
    return pd.read_csv(file, low_memory=False)

if not uploaded:
    st.info("Upload a CSV to start exploring. Example datasets: Titanic, Iris, or any exported spreadsheet.")
    st.stop()

# Load dataframe
try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# ----------------------
# Basic dataset info
# ----------------------
with st.expander("🔎 Dataset overview", expanded=True):
    cols = df.columns.tolist()
    st.write(f"**Rows:** {df.shape[0]:,}     •     **Columns:** {df.shape[1]}")
    st.write("**Columns:**")
    st.write(cols)

    # data types and missing summary
    dtype_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum().values,
        "missing": df.isna().sum().values,
        "missing_%": (df.isna().mean().values * 100).round(2)
    })
    st.dataframe(dtype_df, use_container_width=True)

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.header("Filter & View options")

# Select subset of columns to display
default_cols = cols if len(cols) <= 8 else cols[:8]
selected_cols = st.sidebar.multiselect("Columns to display", options=cols, default=default_cols)

# Quick text search across dataframe (applies only to displayed columns)
search_text = st.sidebar.text_input("Search (text, across selected columns)")

# Sampling & head/tail
sample_mode = st.sidebar.selectbox("Rows to show", ["Head (first 50)", "Tail (last 50)", "Sample (random)"])
sample_n = st.sidebar.number_input("Rows to return (for Sample)", value=50, min_value=1, max_value=10000, step=1)

# Numeric filtering: automatically find numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# Multi-filter widgets for numeric columns (show up to 3 by default)
if numeric_cols:
    st.sidebar.subheader("Numeric filters (range)")
    numeric_filters = {}
    # Show up to first 3 numeric columns with sliders to avoid sidebar overflow
    for col in numeric_cols[:6]:
        lo = float(df[col].min(skipna=True)) if pd.notna(df[col].min(skipna=True)) else 0.0
        hi = float(df[col].max(skipna=True)) if pd.notna(df[col].max(skipna=True)) else 1.0
        if lo == hi:
            # constant column
            numeric_filters[col] = (lo, hi)
            st.sidebar.write(f"{col}: constant ({lo})")
        else:
            rng = st.sidebar.slider(f"{col}", min_value=lo, max_value=hi, value=(lo, hi))
            numeric_filters[col] = rng
else:
    numeric_filters = {}

# Categorical filters (multi-select)
cat_filters = {}
if cat_cols:
    st.sidebar.subheader("Categorical filters")
    for col in cat_cols[:6]:  # limit to first 6 to keep UI tidy
        uniques = df[col].dropna().unique().tolist()
        if len(uniques) <= 50:
            sel = st.sidebar.multiselect(f"{col}", options=sorted(uniques), default=uniques)
            cat_filters[col] = sel
        else:
            # too many unique categories: provide text filter
            text = st.sidebar.text_input(f"Filter values in {col} (comma-separated)")
            if text:
                wanted = [v.strip() for v in text.split(",") if v.strip()]
                cat_filters[col] = wanted
            else:
                cat_filters[col] = None

# ----------------------
# Apply filtering
# ----------------------
filtered = df.copy()

# apply numeric filters
for col, (low, high) in numeric_filters.items():
    if col in filtered.columns:
        filtered = filtered[filtered[col].between(low, high)]

# apply categorical filters
for col, sel in cat_filters.items():
    if sel is None or col not in filtered.columns:
        continue
    if len(sel) > 0:
        filtered = filtered[filtered[col].isin(sel)]

# apply search across selected columns
if search_text and selected_cols:
    mask = pd.Series(False, index=filtered.index)
    for c in selected_cols:
        # convert to string and do case-insensitive contains
        mask = mask | filtered[c].astype(str).str.contains(search_text, case=False, na=False)
    filtered = filtered[mask]

# ----------------------
# Main layout: data + quick visuals
# ----------------------
st.markdown("## 📁 Data preview")

left, right = st.columns([3, 2])

with left:
    # Show the portion selected by sample mode
    if sample_mode.startswith("Head"):
        view_df = filtered.head(sample_n)
    elif sample_mode.startswith("Tail"):
        view_df = filtered.tail(sample_n)
    else:
        view_df = filtered.sample(n=min(sample_n, len(filtered)), random_state=42)

    # Show only selected columns if any chosen
    if selected_cols:
        view_df = view_df[selected_cols]

    st.dataframe(view_df, use_container_width=True)

    st.write(f"Showing **{len(view_df):,}** rows (filtered from {len(filtered):,} rows — original {len(df):,}).")

    # Download filtered data
    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")

with right:
    st.subheader("Summary")
    st.write("**Basic statistics (selected numeric columns)**")
    if numeric_cols:
        st.write(filtered[numeric_cols].describe().T)
    else:
        st.write("No numeric columns found.")

    # Missing values summary
    st.write("---")
    st.write("**Missing values**")
    missing = (filtered.isna().sum() / len(filtered) * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.table(missing.to_frame("missing_%"))
    else:
        st.write("No missing values in filtered data.")

# ----------------------
# Visualizations
# ----------------------
st.markdown("## 📈 Quick visualizations")
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.write("### Histogram (choose a numeric column)")
    hist_col = st.selectbox("Numeric column for histogram", options=[None] + numeric_cols, index=0)
    if hist_col:
        fig, ax = plt.subplots()
        ax.hist(filtered[hist_col].dropna(), bins=30)
        ax.set_title(f"Histogram — {hist_col}")
        ax.set_xlabel(hist_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

with viz_col2:
    st.write("### Scatter (choose X & Y)")
    scatter_x = st.selectbox("X axis", options=[None] + numeric_cols, index=0, key="sx")
    scatter_y = st.selectbox("Y axis", options=[None] + numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="sy")
    color_col = st.selectbox("Color by (categorical)", options=[None] + cat_cols, index=0, key="sc")
    if scatter_x and scatter_y:
        fig, ax = plt.subplots()
        if color_col:
            # plot each category separately for simple coloring
            for cat, sub in filtered.groupby(color_col):
                ax.scatter(sub[scatter_x], sub[scatter_y], label=str(cat), alpha=0.7, s=20)
            ax.legend(title=color_col, bbox_to_anchor=(1.0, 1.0))
        else:
            ax.scatter(filtered[scatter_x], filtered[scatter_y], s=20, alpha=0.7)
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        ax.set_title(f"{scatter_y} vs {scatter_x}")
        st.pyplot(fig)

# ----------------------
# Extras: correlation heatmap (if enough numeric columns)
# ----------------------
if len(numeric_cols) >= 2:
    st.markdown("## 🔗 Correlation (numeric columns)")
    corr = filtered[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)

# ----------------------
# Footer / tips
# ----------------------
st.markdown("---")
st.caption("Tips: Use the sidebar to filter columns quickly. For large files (>10k rows) prefer sampling or column selection to keep the UI responsive.")
