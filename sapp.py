"""
Global Trade & Risk Intelligence - Streamlit Dashboard

HOW TO RUN:
Open Anaconda Prompt and type:

cd "C:/Users/Windows 10/GL/My projects Practice/global-trade-risk-intelligence/notebooks"

streamlit run app.py

By-Rasika 
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
warnings.filterwarnings('ignore')

# Your exact project path
BASE = r"data/raw/msr.csv"

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Global Trade & Risk Intelligence",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* =====================================================
   MAIN APP BACKGROUND
===================================================== */
[data-testid="stAppViewContainer"] {
    background-color: #0B1120;
}
[data-testid="stMain"] {
    background-color: #0B1120;
}

/* =====================================================
   SIDEBAR BACKGROUND
===================================================== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827, #1E293B);
    border-right: 2px solid #334155;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #111827, #1E293B);
}

/* =====================================================
   ALL SIDEBAR TEXT
===================================================== */
[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
    opacity: 1 !important;
}

/* =====================================================
   SIDEBAR TITLE
===================================================== */
[data-testid="stSidebar"] h1 {
    color: #60A5FA !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

/* =====================================================
   LABELS (Year, Risk, Countries, Navigate)
===================================================== */
[data-testid="stWidgetLabel"] {
    color: #E2E8F0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* =====================================================
   CAPTIONS
===================================================== */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* =====================================================
   SLIDER
===================================================== */
.stSlider p {
    color: #93C5FD !important;
    font-weight: 600 !important;
}
.stSlider [data-baseweb="slider"] div {
    background-color: #1e293b !important;
}

/* =====================================================
   SELECTBOX / MULTISELECT
===================================================== */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
}
.stSelectbox div[data-baseweb="select"] span,
.stMultiSelect div[data-baseweb="select"] span {
    color: #FFFFFF !important;
}
/* Dropdown options list */
[data-baseweb="popover"] {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
}
[data-baseweb="menu"] li {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
}
[data-baseweb="menu"] li:hover {
    background-color: #334155 !important;
}

/* =====================================================
   RADIO BUTTONS
===================================================== */
.stRadio label {
    color: #FFFFFF !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    opacity: 1 !important;
    padding: 4px 8px;
    border-radius: 6px;
    transition: background 0.2s;
}
.stRadio label:hover {
    background-color: #334155 !important;
}
/* Selected radio option highlight */
.stRadio [data-testid="stMarkdownContainer"] p {
    color: #60A5FA !important;
    font-weight: 600 !important;
}

/* =====================================================
   DIVIDER
===================================================== */
[data-testid="stSidebar"] hr {
    border-color: #475569 !important;
    margin: 12px 0 !important;
}

/* =====================================================
   MAIN PAGE HEADINGS
===================================================== */
h1 { color: #2E75B6 !important; }
h2 { color: #3498db !important; }
h3 { color: #85C1E9 !important; }

/* =====================================================
   METRICS — Cards
===================================================== */
.stMetric {
    background-color: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}
.stMetric label {
    color: #85C1E9 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
.stMetric [data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 24px !important;
    font-weight: 700 !important;
}
.stMetric [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* =====================================================
   DATAFRAME / TABLES
===================================================== */
[data-testid="stDataFrame"] {
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}
.dataframe thead tr th {
    background-color: #1E3A5F !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
.dataframe tbody tr:nth-child(even) {
    background-color: #1E293B !important;
}
.dataframe tbody tr:nth-child(odd) {
    background-color: #0F172A !important;
}
.dataframe tbody tr td {
    color: #E2E8F0 !important;
}

/* =====================================================
   SUBHEADER TEXT (st.subheader)
===================================================== */
[data-testid="stMarkdownContainer"] h3 {
    color: #60A5FA !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    border-bottom: 1px solid #334155;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

/* =====================================================
   CAPTION TEXT (main area)
===================================================== */
[data-testid="stMarkdownContainer"] small,
.stCaption {
    color: #94A3B8 !important;
    font-size: 13px !important;
}

/* =====================================================
   DIVIDER (main area)
===================================================== */
hr {
    border-color: #334155 !important;
}

/* =====================================================
   SCROLLBAR
===================================================== */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0F172A;
}
::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #64748B;
}

/* =====================================================
   PLOTLY CHART CONTAINER
===================================================== */
.stPlotlyChart {
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 4px;
    background-color: #0F172A;
}

/* =====================================================
   MULTISELECT TAGS
===================================================== */
[data-baseweb="tag"] {
    background-color: #1E3A5F !important;
    border: 1px solid #2E75B6 !important;
    border-radius: 4px !important;
}
[data-baseweb="tag"] span {
    color: #FFFFFF !important;
}

/* =====================================================
   BUTTON (if any)
===================================================== */
.stButton button {
    background-color: #1E3A5F !important;
    color: #FFFFFF !important;
    border: 1px solid #2E75B6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton button:hover {
    background-color: #2E75B6 !important;
    border-color: #60A5FA !important;
}
</style>
""", unsafe_allow_html=True)

# ── Risk Colors ───────────────────────────────────────────────
RISK_COLORS = {
    'Low'     : '#2ecc71',
    'Moderate': '#f39c12',
    'High'    : '#e67e22',
    'Extreme' : '#e74c3c',
    'Unknown' : '#95a5a6',
}

# ── Plotly Theme ──────────────────────────────────────────────
THEME = dict(
    plot_bgcolor  = 'rgba(0,0,0,0)',
    paper_bgcolor = 'rgba(0,0,0,0)',
    font          = dict(color='white', family='Arial'),
)

# ── Load Data ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = os.path.join(
        BASE, "data", "processed", "master_dataset_v3.csv"
    )
    if not os.path.exists(csv_path):
        st.error(f"File not found:\n{csv_path}")
        st.stop()

    df = pd.read_csv(csv_path)
    df['year']      = df['year'].astype(int)
    df['iso3_code'] = df['iso3_code'].astype(str).str.upper()

    # Add operational KPIs if not already in file
    if 'order_fulfillment_rate' not in df.columns:
        df['order_fulfillment_rate'] = (
            df['lpi_timeliness_score']
            .fillna(df['lpi_overall_score'])
            .fillna(3.0).clip(1, 5) / 5 * 100
        ).round(2)

    if 'inventory_turnover' not in df.columns:
        df['inventory_turnover'] = (
            df['trade_in_goods_percent_gdp']
            .fillna(df['trade_percent_gdp'])
            .fillna(50).clip(0, 200) / 10
        ).round(2)

    if 'lead_time_variability_pct' not in df.columns:
        df['lead_time_variability_pct'] = (
            df['logistics_stress'].fillna(5).clip(0, 20) * 5
        ).round(2)

    if 'perfect_order_rate' not in df.columns:
        df['perfect_order_rate'] = (
            df['lpi_overall_score']
            .fillna(3.0).clip(1, 5) / 5 * 100
        ).round(2)

    if 'cost_to_serve_index' not in df.columns:
        customs = (
            5 - df['lpi_customs_score'].fillna(3.0).clip(1, 5)
        ) / 4 * 50
        econ = df['economic_vulnerability'].fillna(10).clip(0, 100)
        df['cost_to_serve_index'] = (
            customs * 0.5 + econ * 0.5
        ).round(2)

    if 'composite_risk_index' not in df.columns:
        df['composite_risk_index'] = df.get(
            'composite_risk_index_v3',
            df['gpr_mean'].fillna(100) / 10
        )

    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🌐 Filters")
    st.divider()

    year_min = int(df['year'].min())
    year_max = int(df['year'].max())
    year_range = st.slider(
        "📅 Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

    risk_opts = ['All'] + sorted(
        df['risk_category'].dropna().unique().tolist()
    )
    sel_risk = st.selectbox("⚠️ Risk Category", risk_opts)

    sel_countries = st.multiselect(
        "🌍 Countries (empty = all)",
        sorted(df['iso3_code'].unique().tolist())
    )

    st.divider()
    st.caption(f"📊 Rows     : {len(df):,}")
    st.caption(f"🌍 Countries: {df['iso3_code'].nunique()}")
    st.caption(f"📅 Years    : {year_min}–{year_max}")
    st.caption(f"📐 Features : {len(df.columns)}")
    st.divider()

    page = st.radio("📄 Navigate", [
        "🏠 Executive Overview",
        "📦 Trade Analysis",
        "⚙️ Operational KPIs",
        "🌍 Geopolitical Risk",
        "🔍 Country Deep Dive",
    ])

# ── Apply Filters ─────────────────────────────────────────────
fdf = df[
    (df['year'] >= year_range[0]) &
    (df['year'] <= year_range[1])
].copy()
if sel_risk != 'All':
    fdf = fdf[fdf['risk_category'] == sel_risk]
if sel_countries:
    fdf = fdf[fdf['iso3_code'].isin(sel_countries)]

# ═════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.title("🌐 Global Trade & Risk Intelligence")
    st.caption("Executive Overview — Risk, Trade, Logistics & Geopolitics")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    avg_risk  = fdf['composite_risk_index'].mean()
    n_ctry    = fdf['iso3_code'].nunique()
    avg_lpi   = fdf['lpi_overall_score'].mean()
    avg_gpr   = fdf['gpr_mean'].mean()
    high_risk = (fdf['high_risk_flag'] == 1).sum()

    with c1:
        st.metric("🎯 Avg Risk Index",
                  f"{avg_risk:.1f}/100",
                  f"{avg_risk-df['composite_risk_index'].mean():+.1f} vs all",
                  delta_color="inverse")
    with c2:
        st.metric("🌍 Countries", f"{n_ctry}")
    with c3:
        st.metric("🚢 Avg LPI", f"{avg_lpi:.2f}/5.0")
    with c4:
        st.metric("⚠️ GPR Index", f"{avg_gpr:.1f}",
                  "Baseline=100", delta_color="off")
    with c5:
        st.metric("🔴 High Risk",
                  f"{high_risk}",
                  f"{high_risk/max(len(fdf),1)*100:.1f}%",
                  delta_color="inverse")

    st.divider()
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🗺️ Global Risk Map")
        map_df = (fdf.groupby('iso3_code')['composite_risk_index']
                  .mean().reset_index())
        map_df.columns = ['iso3_code', 'risk']
        fig = px.choropleth(
            map_df, locations='iso3_code', color='risk',
            hover_name='iso3_code',
            color_continuous_scale=[
                [0.00, '#2ecc71'], [0.25, '#f1c40f'],
                [0.50, '#e67e22'], [0.75, '#e74c3c'],
                [1.00, '#922b21'],
            ],
            range_color=[0, 60],
            labels={'risk': 'Risk Index'},
        )
        fig.update_layout(
            **THEME, height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            geo=dict(
                showframe=False, showcoastlines=True,
                coastlinecolor='#374151',
                bgcolor='rgba(0,0,0,0)',
                landcolor='#1F2937',
                showocean=True, oceancolor='#0E1117',
            ),
            coloraxis_colorbar=dict(
                title='Risk',
                tickvals=[0, 25, 45, 65],
                ticktext=['Low', 'Moderate', 'High', 'Extreme'],
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Risk Distribution")
        dist = fdf['risk_category'].value_counts().reset_index()
        dist.columns = ['category', 'count']
        fig2 = go.Figure(go.Pie(
            labels=dist['category'], values=dist['count'],
            hole=0.55,
            marker_colors=[
                RISK_COLORS.get(c, '#95a5a6')
                for c in dist['category']
            ],
            textinfo='label+percent',
            textfont=dict(color='white', size=12),
        ))
        fig2.update_layout(
            **THEME, showlegend=False, height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(
                text=f"{len(fdf)}<br>obs",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=15, color='white')
            )]
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("**Risk Breakdown**")
        for cat in ['Low', 'Moderate', 'High', 'Extreme']:
            n   = (fdf['risk_category'] == cat).sum()
            pct = n / max(len(fdf), 1) * 100
            c   = RISK_COLORS.get(cat, 'gray')
            st.markdown(
                f"<span style='color:{c}'>●</span> "
                f"**{cat}**: {n} ({pct:.1f}%)",
                unsafe_allow_html=True
            )

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📈 Risk Trend 2010–2023")
        trend = (fdf.groupby('year')['composite_risk_index']
                 .agg(['mean', 'min', 'max']).reset_index())
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=trend['year'], y=trend['max'],
            mode='lines',
            line=dict(color='rgba(0,0,0,0)', width=0),
            showlegend=False
        ))
        fig3.add_trace(go.Scatter(
            x=trend['year'], y=trend['min'],
            fill='tonexty', mode='lines',
            fillcolor='rgba(231,76,60,0.1)',
            line=dict(color='rgba(0,0,0,0)', width=0),
            name='Range'
        ))
        fig3.add_trace(go.Scatter(
            x=trend['year'], y=trend['mean'],
            mode='lines+markers',
            line=dict(color='#e74c3c', width=2.5),
            marker=dict(size=6), name='Mean Risk'
        ))
        fig3.add_vrect(
            x0=2019.5, x1=2021,
            fillcolor='rgba(255,165,0,0.07)',
            layer='below', line_width=0,
            annotation_text='COVID-19',
            annotation=dict(font=dict(color='orange', size=9))
        )
        fig3.update_layout(
            **THEME, height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#374151'),
            yaxis=dict(gridcolor='#374151',
                       title='Risk Index'),
            legend=dict(orientation='h', x=0, y=-0.25,
                        font=dict(color='white'))
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("🏆 Top 15 Riskiest Countries")
        top = (fdf.groupby('iso3_code')['composite_risk_index']
               .mean().nlargest(15).reset_index())
        top.columns = ['country', 'risk']
        top['color'] = top['risk'].apply(
            lambda x: '#e74c3c' if x > 45
            else '#e67e22' if x > 25 else '#2ecc71'
        )
        fig4 = go.Figure(go.Bar(
            x=top['risk'], y=top['country'],
            orientation='h',
            marker_color=top['color'],
            text=top['risk'].round(1),
            textposition='outside',
            textfont=dict(color='white', size=10),
        ))
        fig4.add_vline(x=45, line_dash='dash',
                       line_color='red', opacity=0.5,
                       annotation_text='High Risk',
                       annotation_font=dict(color='red', size=9))
        fig4.update_yaxes(autorange='reversed')
        fig4.update_layout(
            **THEME, height=300,
            margin=dict(l=0, r=60, t=10, b=0),
            xaxis=dict(gridcolor='#374151',
                       title='Risk Index'),
            yaxis=dict(gridcolor='#374151'),
        )
        st.plotly_chart(fig4, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE 2 — TRADE ANALYSIS
# ═════════════════════════════════════════════════════════════
elif page == "📦 Trade Analysis":
    st.title("📦 Trade Flow Analysis")
    st.caption("Trade Openness, FDI Inflows & Export/Import Patterns")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📊 Trade Openness",
                  f"{fdf['trade_openness'].mean():.1f}%")
    with c2:
        st.metric("📤 Exports % GDP",
                  f"{fdf['exports_pct_gdp'].mean():.1f}%")
    with c3:
        st.metric("💰 Avg FDI Inflows",
                  f"${fdf['fdi_inflows_usd'].mean()/1e9:.1f}B")
    with c4:
        surplus = (fdf['trade_balance_usd'] > 0).mean() * 100
        st.metric("⚖️ Surplus Countries", f"{surplus:.1f}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Trade Openness — Top 20")
        op = (fdf.groupby('iso3_code')['trade_openness']
              .mean().nlargest(20).reset_index())
        fig = px.bar(
            op, x='trade_openness', y='iso3_code',
            orientation='h', color='trade_openness',
            color_continuous_scale='Blues',
            labels={'trade_openness': '% GDP'},
            text=op['trade_openness'].round(1)
        )
        fig.add_vline(x=80, line_dash='dash',
                      line_color='orange', opacity=0.7,
                      annotation_text='80% Threshold',
                      annotation_font=dict(color='orange', size=9))
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(**THEME, height=420,
                          margin=dict(l=0, r=60, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Trade Trend 2010–2023")
        tr = fdf.groupby('year').agg(
            openness=('trade_openness', 'mean'),
            exports=('exports_pct_gdp', 'mean'),
            imports=('imports_pct_gdp', 'mean'),
        ).reset_index()
        fig2 = go.Figure()
        for col_n, color, lbl in [
            ('openness', '#3498db', 'Trade Openness'),
            ('exports',  '#2ecc71', 'Exports % GDP'),
            ('imports',  '#e74c3c', 'Imports % GDP'),
        ]:
            fig2.add_trace(go.Scatter(
                x=tr['year'], y=tr[col_n],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=5), name=lbl
            ))
        fig2.update_layout(
            **THEME, height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#374151'),
            yaxis=dict(gridcolor='#374151', title='% of GDP'),
            legend=dict(orientation='h', x=0, y=-0.2,
                        font=dict(color='white'))
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("💰 FDI Inflows — Top 15")
        fdi = (fdf.groupby('iso3_code')['fdi_inflows_usd']
               .mean().nlargest(15).reset_index())
        fdi['fdi_b'] = fdi['fdi_inflows_usd'] / 1e9
        fig3 = px.bar(
            fdi, x='fdi_b', y='iso3_code', orientation='h',
            color='fdi_b', color_continuous_scale='Purples',
            labels={'fdi_b': 'Billion USD'},
            text=fdi['fdi_b'].round(1)
        )
        fig3.update_yaxes(autorange='reversed')
        fig3.update_layout(**THEME, height=380,
                           margin=dict(l=0, r=60, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("📊 Exports vs Imports")
        sc = fdf.dropna(subset=[
            'exports_pct_gdp', 'imports_pct_gdp', 'gdp_current_usd'
        ])
        fig4 = px.scatter(
            sc, x='exports_pct_gdp', y='imports_pct_gdp',
            color='risk_category', size='gdp_current_usd',
            size_max=30, hover_name='iso3_code',
            color_discrete_map=RISK_COLORS, opacity=0.7,
            labels={'exports_pct_gdp': 'Exports % GDP',
                    'imports_pct_gdp': 'Imports % GDP'},
        )
        fig4.add_shape(
            type='line', x0=0, y0=0, x1=200, y1=200,
            line=dict(color='gray', dash='dash', width=1)
        )
        fig4.update_layout(
            **THEME, height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#374151'),
            yaxis=dict(gridcolor='#374151'),
            legend=dict(font=dict(color='white'),
                        title=dict(font=dict(color='white')))
        )
        st.plotly_chart(fig4, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE 3 — OPERATIONAL KPIs
# ═════════════════════════════════════════════════════════════
elif page == "⚙️ Operational KPIs":
    st.title("⚙️ Operational KPI Dashboard")
    st.caption("Order Fulfillment · Inventory · Lead Time · Perfect Order · Cost-to-Serve")
    st.divider()

    def make_gauge(value, title, min_v, max_v, target, color):
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=round(value, 1),
            title={'text': title,
                   'font': {'size': 13, 'color': 'white'}},
            delta={'reference': target,
                   'increasing': {'color': '#2ecc71'},
                   'decreasing': {'color': '#e74c3c'}},
            gauge={
                'axis': {'range': [min_v, max_v],
                         'tickcolor': 'white',
                         'tickfont': {'color': 'white', 'size': 9}},
                'bar': {'color': color},
                'bgcolor': '#1F2937',
                'bordercolor': '#374151',
                'threshold': {
                    'line': {'color': 'white', 'width': 2},
                    'thickness': 0.75, 'value': target,
                },
                'steps': [
                    {'range': [min_v, target*0.9], 'color': '#450a0a'},
                    {'range': [target*0.9, target], 'color': '#7f1d1d'},
                    {'range': [target, max_v],      'color': '#14532d'},
                ],
            },
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=210, margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig

    g1, g2, g3, g4, g5 = st.columns(5)
    ofr = fdf['order_fulfillment_rate'].mean()
    inv = fdf['inventory_turnover'].mean()
    ltv = fdf['lead_time_variability_pct'].mean()
    por = fdf['perfect_order_rate'].mean()
    cts = fdf['cost_to_serve_index'].mean()

    with g1:
        st.plotly_chart(make_gauge(ofr, 'Order Fulfillment %',
                                   0, 100, 95, '#3498db'),
                        use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(inv, 'Inventory Turnover x',
                                   0, 15, 5, '#2ecc71'),
                        use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(ltv, 'Lead Time Var %',
                                   0, 50, 10, '#e67e22'),
                        use_container_width=True)
    with g4:
        st.plotly_chart(make_gauge(por, 'Perfect Order Rate %',
                                   0, 100, 98, '#9b59b6'),
                        use_container_width=True)
    with g5:
        st.plotly_chart(make_gauge(cts, 'Cost-to-Serve Index',
                                   0, 60, 25, '#e74c3c'),
                        use_container_width=True)

    st.divider()
    st.subheader("📋 KPI Status Summary")
    st.dataframe(pd.DataFrame({
        'KPI'        : ['Order Fulfillment Rate', 'Inventory Turnover',
                        'Lead Time Variability', 'Perfect Order Rate',
                        'Cost-to-Serve Index'],
        'Current'    : [f"{ofr:.1f}%", f"{inv:.2f}x",
                        f"{ltv:.1f}%", f"{por:.1f}%", f"{cts:.2f}"],
        'Target'     : ['> 95%', '5-10x', '< 10%', '> 98%', 'Minimize'],
        'Status'     : [
            '🟡 Watch' if ofr < 95 else '🟢 Good',
            '🟢 Good'  if 5 <= inv <= 10 else '🟡 Watch',
            '🟢 Good'  if ltv < 10 else '🔴 Alert',
            '🔴 Alert' if por < 75 else '🟡 Watch',
            '🟡 Monitor',
        ],
        'Stakeholder': ['Operations Manager', 'Supply Chain Director',
                        'Procurement Head', 'Customer Service Head',
                        'CFO / Finance'],
    }), use_container_width=True, hide_index=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Order Fulfillment by Country")
        ofr_c = (fdf.groupby('iso3_code')['order_fulfillment_rate']
                 .mean().nlargest(15).reset_index())
        ofr_c.columns = ['country', 'ofr']
        ofr_c['color'] = ofr_c['ofr'].apply(
            lambda x: '#2ecc71' if x >= 80
            else '#f39c12' if x >= 65 else '#e74c3c'
        )
        fig = go.Figure(go.Bar(
            x=ofr_c['ofr'], y=ofr_c['country'],
            orientation='h', marker_color=ofr_c['color'],
            text=ofr_c['ofr'].round(1), textposition='outside',
            textfont=dict(color='white', size=10),
        ))
        fig.add_vline(x=95, line_dash='dash', line_color='green',
                      opacity=0.7, annotation_text='Target: 95%',
                      annotation_font=dict(color='green', size=9))
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(**THEME, height=380,
                          margin=dict(l=0, r=60, t=10, b=0),
                          xaxis=dict(gridcolor='#374151'),
                          yaxis=dict(gridcolor='#374151'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💸 Cost-to-Serve — Highest Markets")
        cts_c = (fdf.groupby('iso3_code')['cost_to_serve_index']
                 .mean().nlargest(15).reset_index())
        cts_c.columns = ['country', 'cts']
        fig2 = px.bar(
            cts_c, x='cts', y='country', orientation='h',
            color='cts', color_continuous_scale='Reds',
            text=cts_c['cts'].round(1),
            labels={'cts': 'Cost-to-Serve Index'},
        )
        fig2.update_yaxes(autorange='reversed')
        fig2.update_layout(**THEME, height=380,
                           margin=dict(l=0, r=60, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE 4 — GEOPOLITICAL RISK
# ═════════════════════════════════════════════════════════════
elif page == "🌍 Geopolitical Risk":
    st.title("🌍 Geopolitical Risk Monitor")
    st.caption("GPR Index · Political Stability · Military Expenditure")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("⚠️ Avg GPR Index",
                  f"{fdf['gpr_mean'].mean():.1f}",
                  "Baseline: 100", delta_color="off")
    with c2:
        st.metric("📈 Max GPR Spike", f"{fdf['gpr_max'].max():.0f}")
    with c3:
        st.metric("🏛️ Political Stability",
                  f"{fdf['political_stability_score'].mean():.2f}",
                  "Scale: -2.5 to +2.5", delta_color="off")
    with c4:
        st.metric("🪖 Military Exp",
                  f"{fdf['military_expenditure_pct_gdp'].mean():.2f}% GDP")

    st.divider()
    st.subheader("📈 GPR Index Timeline 2010–2023")
    gpr_t = fdf.groupby('year').agg(
        mean_gpr=('gpr_mean', 'mean'),
        max_gpr =('gpr_max',  'mean'),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gpr_t['year'], y=gpr_t['max_gpr'],
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=gpr_t['year'], y=gpr_t['mean_gpr'],
        fill='tonexty', mode='lines+markers',
        fillcolor='rgba(231,76,60,0.1)',
        line=dict(color='#e74c3c', width=2.5),
        marker=dict(size=7), name='Mean GPR'
    ))
    fig.add_hline(y=100, line_dash='dash', line_color='gray',
                  annotation_text='Baseline (100)',
                  annotation_font=dict(color='gray', size=9))
    fig.add_hline(y=200, line_dash='dot', line_color='orange',
                  annotation_text='Elevated Risk (200)',
                  annotation_font=dict(color='orange', size=9))

    for yr, lbl in {2015: 'ISIS/Syria', 2016: 'Brexit',
                    2018: 'Trade War', 2020: 'COVID-19',
                    2022: 'Russia-Ukraine'}.items():
        if yr in gpr_t['year'].values:
            val = gpr_t[gpr_t['year'] == yr]['mean_gpr'].values[0]
            fig.add_annotation(
                x=yr, y=val + 4, text=lbl,
                showarrow=True, arrowhead=2,
                arrowcolor='white',
                font=dict(color='white', size=9),
                ax=0, ay=-28
            )
    fig.update_layout(
        **THEME, height=350,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor='#374151', title='Year'),
        yaxis=dict(gridcolor='#374151', title='GPR Index'),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏛️ Political Stability Map")
        pol_df = (fdf.groupby('iso3_code')['political_stability_score']
                  .mean().reset_index())
        pol_df.columns = ['iso3_code', 'stability']
        fig2 = px.choropleth(
            pol_df, locations='iso3_code', color='stability',
            hover_name='iso3_code',
            color_continuous_scale=[
                [0.0, '#e74c3c'],
                [0.5, '#f1c40f'],
                [1.0, '#2ecc71'],
            ],
            range_color=[-2.5, 2.5],
            labels={'stability': 'WGI Score'},
        )
        fig2.update_layout(
            **THEME, height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            geo=dict(
                showframe=False, showcoastlines=True,
                coastlinecolor='#374151',
                bgcolor='rgba(0,0,0,0)',
                landcolor='#1F2937',
                showocean=True, oceancolor='#0E1117',
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("🪖 Military Expenditure % GDP")
        mil_df = (fdf.groupby('iso3_code')
                  ['military_expenditure_pct_gdp']
                  .mean().dropna().nlargest(15).reset_index())
        mil_df.columns = ['country', 'mil']
        fig3 = px.bar(
            mil_df, x='mil', y='country', orientation='h',
            color='mil', color_continuous_scale='Reds',
            text=mil_df['mil'].round(2),
            labels={'mil': '% of GDP'},
        )
        fig3.add_vline(x=2, line_dash='dash',
                       line_color='yellow', opacity=0.7,
                       annotation_text='NATO: 2%',
                       annotation_font=dict(color='yellow', size=9))
        fig3.update_yaxes(autorange='reversed')
        fig3.update_layout(**THEME, height=300,
                           margin=dict(l=0, r=60, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE 5 — COUNTRY DEEP DIVE
# ═════════════════════════════════════════════════════════════
elif page == "🔍 Country Deep Dive":
    st.title("🔍 Country Deep Dive")
    st.caption("Complete KPI analysis for any single country")
    st.divider()

    default_idx = (
        list(sorted(df['iso3_code'].unique())).index('USA')
        if 'USA' in df['iso3_code'].values else 0
    )
    country = st.selectbox(
        "🌍 Select Country",
        sorted(df['iso3_code'].unique()),
        index=default_idx
    )

    cdf = df[df['iso3_code'] == country].sort_values('year')
    if len(cdf) == 0:
        st.warning("No data for this country.")
        st.stop()

    latest = cdf.iloc[-1]
    st.subheader(
        f"📊 {country} — Latest Year: {int(latest['year'])}"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("🎯 Risk Index",
                  f"{latest.get('composite_risk_index', 0):.1f}")
    with c2:
        st.metric("📊 Trade Openness",
                  f"{latest.get('trade_openness', 0):.1f}%")
    with c3:
        st.metric("🚢 LPI Score",
                  f"{latest.get('lpi_overall_score', 0):.2f}/5")
    with c4:
        st.metric("💹 GDP Growth",
                  f"{latest.get('gdp_growth_rate_pct', 0):.1f}%")
    with c5:
        st.metric("⚠️ Risk Category",
                  str(latest.get('risk_category', 'N/A')))

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Risk Index History")
        risk_col = ('composite_risk_index'
                    if 'composite_risk_index' in cdf.columns
                    else 'composite_risk_index_v3')
        fig = go.Figure(go.Scatter(
            x=cdf['year'], y=cdf[risk_col],
            mode='lines+markers', fill='tozeroy',
            fillcolor='rgba(231,76,60,0.1)',
            line=dict(color='#e74c3c', width=2.5),
            marker=dict(size=8), name='Risk Index'
        ))
        fig.add_hline(y=45, line_dash='dash',
                      line_color='orange',
                      annotation_text='High Risk',
                      annotation_font=dict(color='orange', size=9))
        fig.update_layout(
            **THEME, height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(gridcolor='#374151', title='Year'),
            yaxis=dict(gridcolor='#374151', title='Risk Index'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 KPI Radar Chart")
        radar = {
            'Order Fulfillment': cdf['order_fulfillment_rate'].mean(),
            'Perfect Order'    : cdf['perfect_order_rate'].mean(),
            'LPI Score'        : cdf['lpi_overall_score'].mean()/5*100,
            'Trade Openness'   : min(cdf['trade_openness'].mean(), 100),
            'GDP Growth'       : min(max(
                cdf['gdp_growth_rate_pct'].mean()*5+50, 0), 100),
        }
        cats = list(radar.keys())
        vals = list(radar.values())
        fig2 = go.Figure(go.Scatterpolar(
            r=vals+[vals[0]], theta=cats+[cats[0]],
            fill='toself',
            fillcolor='rgba(52,152,219,0.2)',
            line=dict(color='#3498db', width=2),
            name=country
        ))
        fig2.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100],
                                gridcolor='#374151',
                                tickfont=dict(color='white', size=9)),
                angularaxis=dict(gridcolor='#374151',
                                 tickfont=dict(color='white', size=11))
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False, height=300,
            margin=dict(l=40, r=40, t=20, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 Complete KPI History")
    show = [c for c in [
        'year', 'composite_risk_index', 'risk_category',
        'trade_openness', 'inflation_rate_pct',
        'gdp_growth_rate_pct', 'lpi_overall_score',
        'gpr_mean', 'unemployment_rate_pct',
        'order_fulfillment_rate', 'perfect_order_rate',
        'cost_to_serve_index',
    ] if c in cdf.columns]
    st.dataframe(
        cdf[show].round(2).sort_values('year', ascending=False),
        use_container_width=True, hide_index=True
    )

# ── Footer ────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("🌐 Global Trade & Risk Intelligence")
st.sidebar.caption("Phase 1 — Data Analytics Dashboard")
st.sidebar.caption("Built with Streamlit + Plotly")
