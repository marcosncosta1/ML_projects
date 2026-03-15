"""
Switzerland Avalanche Risk Dashboard  —  Enhanced
Data: WSL Institute for Snow and Avalanche Research SLF · CC BY 4.0
"""

import re
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pyproj import Transformer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏔️ Swiss Avalanche Dashboard",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Compact metric cards */
    [data-testid="metric-container"] { background:#f0f4f8; border-radius:8px; padding:10px 14px; }
    /* Webcam cards */
    .webcam-label { font-size:0.78rem; color:#6b7280; margin-top:4px; text-align:center; }
    .resort-header { font-size:1.1rem; font-weight:700; margin-bottom:2px; }
    .resort-meta { font-size:0.82rem; color:#6b7280; margin-bottom:10px; }
    /* Danger scale badges */
    .d1{background:#61c050;color:#fff;padding:3px 10px;border-radius:12px;font-weight:600;}
    .d2{background:#f5d63c;color:#333;padding:3px 10px;border-radius:12px;font-weight:600;}
    .d3{background:#f47f21;color:#fff;padding:3px 10px;border-radius:12px;font-weight:600;}
    .d4{background:#e02020;color:#fff;padding:3px 10px;border-radius:12px;font-weight:600;}
    .d5{background:#1a0a0a;color:#fff;padding:3px 10px;border-radius:12px;font-weight:600;}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────
SLF_URL = (
    "https://www.slf.ch/en/avalanches/avalanches-and-avalanche-accidents/"
    "all-reported-avalanche-accidents-in-current-year/"
)

DANGER_COLORS = {
    "1": "#61c050", "2": "#f5d63c", "3": "#f47f21", "4": "#e02020", "5": "#1a0a0a",
    "Low": "#61c050", "Moderate": "#f5d63c", "Considerable": "#f47f21",
    "High": "#e02020", "Very High": "#1a0a0a",
}

@st.cache_data(ttl=3600, show_spinner="Fetching latest SLF accident data…")
def load_accidents() -> pd.DataFrame:
    resp = requests.get(SLF_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    rows = []
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()

    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue
        link = tds[2].find("a")
        lat, lon = None, None
        if link and "center=" in link.get("href", ""):
            m = re.search(r"center=([\d.]+),([\d.]+)", link["href"])
            if m:
                lon, lat = transformer.transform(float(m.group(1)), float(m.group(2)))

        activity_map = {
            "1": "Backcountry", "2": "Off-piste",
            "3": "Transportation", "4": "Buildings",
        }
        rows.append({
            "date":      tds[0].get_text(strip=True),
            "canton":    tds[1].get_text(strip=True),
            "location":  tds[2].get_text(strip=True),
            "elevation": int(tds[3].get_text(strip=True)) if tds[3].get_text(strip=True).isdigit() else None,
            "aspect":    tds[4].get_text(strip=True),
            "activity":  activity_map.get(tds[5].get_text(strip=True), tds[5].get_text(strip=True)),
            "danger":    tds[6].get_text(strip=True),
            "caught":    int(tds[7].get_text(strip=True) or 0),
            "buried":    int(tds[8].get_text(strip=True) or 0),
            "killed":    int(tds[9].get_text(strip=True) or 0),
            "lat": lat, "lon": lon,
        })

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["week"]  = df["date"].dt.to_period("W").astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df.dropna(subset=["lat", "lon"])


def row_color(r):
    if r["killed"] > 0:   return [220, 38, 38, 220]
    if r["buried"] > 0:   return [249, 115, 22, 210]
    return [59, 130, 246, 180]


# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = load_accidents()
except Exception as e:
    st.error(f"Could not load SLF data: {e}")
    st.stop()

if df.empty:
    st.warning("No accident data found — the SLF page structure may have changed.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.slf.ch/fileadmin/user_upload/SLF_Logo_RGB_pos.svg",
             width=130)
    st.markdown("### 🔍 Filters")

    cantons = ["All"] + sorted(df["canton"].dropna().unique().tolist())
    sel_canton = st.selectbox("Canton", cantons)

    activities = ["All"] + sorted(df["activity"].dropna().unique().tolist())
    sel_activity = st.selectbox("Activity", activities)

    casualty_filter = st.radio(
        "Casualties",
        ["All", "With fatalities", "No fatalities"],
    )

    st.markdown("---")
    st.metric("Total accidents", len(df))
    st.metric("Persons caught",  int(df["caught"].sum()))
    st.metric("Buried",          int(df["buried"].sum()))
    st.metric("Fatalities",      int(df["killed"].sum()))
    st.markdown("---")
    st.caption(
        "Data: [WSL SLF](https://www.slf.ch) · CC BY 4.0\n\n"
        f"Last refreshed: {datetime.now().strftime('%d %b %Y %H:%M')}"
    )

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_canton != "All":
    filtered = filtered[filtered["canton"] == sel_canton]
if sel_activity != "All":
    filtered = filtered[filtered["activity"] == sel_activity]
if casualty_filter == "With fatalities":
    filtered = filtered[filtered["killed"] > 0]
elif casualty_filter == "No fatalities":
    filtered = filtered[filtered["killed"] == 0]

filtered = filtered.copy().reset_index(drop=True)
filtered["color"] = filtered.apply(row_color, axis=1)
filtered["size"]  = (filtered["caught"] * 40).clip(lower=30)
filtered["tooltip"] = filtered.apply(
    lambda r: f"{r['location']} ({r['canton']})\n"
              f"Date: {r['date'].strftime('%d %b %Y') if pd.notna(r['date']) else '?'}\n"
              f"Caught: {r['caught']} · Buried: {r['buried']} · Killed: {r['killed']}\n"
              f"Elevation: {r['elevation']}m · Danger: {r['danger']}",
    axis=1,
)

# ── Swiss centre view ─────────────────────────────────────────────────────────
SWITZERLAND = pdk.ViewState(latitude=46.8, longitude=8.3, zoom=7, pitch=0)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_map, tab_analytics, tab_webcams, tab_about = st.tabs(
    ["🗺️  Map", "📊  Analytics", "📹  Webcams", "ℹ️  About"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader(f"Avalanche accidents — {len(filtered)} event(s) shown")

    map_view = st.radio(
        "Map style",
        ["🔵 Scatter", "🔥 Heatmap", "⬡ Hexagon grid", "🏛️ 3-D columns"],
        horizontal=True,
    )

    col_legend1, col_legend2, col_legend3, _ = st.columns([1, 1, 1, 3])
    col_legend1.markdown("🔵 No burials")
    col_legend2.markdown("🟠 Buried, survived")
    col_legend3.markdown("🔴 Fatal")

    tooltip = {"text": "{tooltip}"}

    if map_view == "🔵 Scatter":
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=filtered,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="size",
            pickable=True,
            opacity=0.85,
            stroked=True,
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=1,
        )

    elif map_view == "🔥 Heatmap":
        # Weight by number of casualties
        filtered["weight"] = (filtered["caught"] + filtered["killed"] * 3).clip(lower=1)
        layer = pdk.Layer(
            "HeatmapLayer",
            data=filtered,
            get_position=["lon", "lat"],
            get_weight="weight",
            radiusPixels=50,
            intensity=1.2,
            threshold=0.05,
        )
        tooltip = None

    elif map_view == "⬡ Hexagon grid":
        layer = pdk.Layer(
            "HexagonLayer",
            data=filtered,
            get_position=["lon", "lat"],
            radius=5000,
            elevation_scale=50,
            elevation_range=[0, 1000],
            extruded=True,
            pickable=True,
            auto_highlight=True,
            coverage=0.85,
        )
        SWITZERLAND = pdk.ViewState(latitude=46.8, longitude=8.3, zoom=7, pitch=40, bearing=0)

    else:  # 3-D columns
        filtered["col_height"] = (filtered["caught"] * 200).clip(lower=100)
        layer = pdk.Layer(
            "ColumnLayer",
            data=filtered,
            get_position=["lon", "lat"],
            get_elevation="col_height",
            elevation_scale=1,
            radius=3000,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        SWITZERLAND = pdk.ViewState(latitude=46.8, longitude=8.3, zoom=7, pitch=45, bearing=-10)

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=SWITZERLAND,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    )
    st.pydeck_chart(deck, use_container_width=True)

    with st.expander("📋 Accident table"):
        display_cols = ["date", "canton", "location", "elevation", "aspect",
                        "activity", "danger", "caught", "buried", "killed"]
        st.dataframe(
            filtered[display_cols].sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "killed": st.column_config.NumberColumn("Killed", format="%d 💀"),
                "buried": st.column_config.NumberColumn("Buried", format="%d ⬇️"),
                "caught": st.column_config.NumberColumn("Caught", format="%d"),
                "elevation": st.column_config.NumberColumn("Elevation (m)"),
            },
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Accidents",       len(filtered))
    k2.metric("Caught",          int(filtered["caught"].sum()))
    k3.metric("Buried",          int(filtered["buried"].sum()))
    k4.metric("Fatalities",      int(filtered["killed"].sum()))
    elev_mean = filtered["elevation"].dropna().mean()
    k5.metric("Avg elevation", f"{elev_mean:.0f} m" if elev_mean else "—")

    st.markdown("---")

    # Timeline
    if not filtered.empty and "week" in filtered.columns:
        timeline = filtered.groupby("week").agg(
            accidents=("date", "count"),
            caught=("caught", "sum"),
            killed=("killed", "sum"),
        ).reset_index()

        fig_time = go.Figure()
        fig_time.add_trace(go.Bar(x=timeline["week"], y=timeline["accidents"],
                                  name="Accidents", marker_color="#3b82f6", opacity=0.7))
        fig_time.add_trace(go.Scatter(x=timeline["week"], y=timeline["killed"],
                                      name="Fatalities", mode="lines+markers",
                                      line=dict(color="#ef4444", width=2),
                                      marker=dict(size=6)))
        fig_time.update_layout(
            title="Accidents & fatalities by week",
            xaxis_title="Week", yaxis_title="Count",
            legend=dict(orientation="h", y=1.1),
            height=320, margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45),
        )
        st.plotly_chart(fig_time, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        canton_counts = (
            filtered.groupby("canton")
            .agg(accidents=("date", "count"), killed=("killed", "sum"))
            .sort_values("accidents", ascending=True)
            .tail(15)
            .reset_index()
        )
        fig_canton = px.bar(
            canton_counts, x="accidents", y="canton", orientation="h",
            color="killed", color_continuous_scale=["#3b82f6", "#ef4444"],
            labels={"accidents": "Accidents", "canton": "Canton", "killed": "Fatalities"},
            title="Accidents by canton (colour = fatalities)",
            height=380,
        )
        fig_canton.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_colorbar=dict(thickness=12, len=0.6),
        )
        st.plotly_chart(fig_canton, use_container_width=True)

    with col_b:
        act_counts = filtered["activity"].value_counts().reset_index()
        act_counts.columns = ["activity", "count"]
        fig_act = px.pie(
            act_counts, values="count", names="activity",
            title="Accidents by activity",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_act.update_layout(
            margin=dict(l=20, r=20, t=50, b=20),
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_act, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        danger_counts = filtered["danger"].value_counts().reset_index()
        danger_counts.columns = ["danger", "count"]
        colors = [DANGER_COLORS.get(str(d), "#94a3b8") for d in danger_counts["danger"]]
        fig_danger = px.bar(
            danger_counts, x="danger", y="count",
            title="Accidents by avalanche danger level",
            color="danger",
            color_discrete_sequence=colors,
            labels={"danger": "Danger level", "count": "Accidents"},
        )
        fig_danger.update_layout(
            showlegend=False, height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_danger, use_container_width=True)

    with col_d:
        elev_data = filtered["elevation"].dropna()
        if not elev_data.empty:
            fig_elev = px.histogram(
                elev_data, nbins=20,
                title="Elevation distribution of accidents",
                labels={"value": "Elevation (m)", "count": "Accidents"},
                color_discrete_sequence=["#6366f1"],
            )
            fig_elev.update_layout(
                height=300, margin=dict(l=20, r=20, t=50, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
            )
            st.plotly_chart(fig_elev, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WEBCAMS
# ═══════════════════════════════════════════════════════════════════════════════

# Webcam data ──────────────────────────────────────────────────────────────────
RESORTS = [
    {
        "name": "🎿 Andermatt-Sedrun",
        "subtitle": "SkiArena Andermatt-Sedrun-Disentis · Uri / Graubünden",
        "meta": "2,963 m top elevation · ~180 km pistes · SLF region: Uri Alps",
        "coords": (46.6357, 8.5939),
        "webcam_page": "https://www.andermatt-sedrun-disentis.ch/andermatt/en/stories/webcams",
        "cams": [
            {
                "label": "Gemsstock (2,917 m)",
                "url": None,
                "link": "https://andermatt-sedrun-disentis.roundshot.com/gemsstock/",
                "note": "360° panoramic cam — opens in new tab",
            },
            {
                "label": "Gütsch (1,849 m)",
                "url": None,
                "link": "https://andermatt-sedrun-disentis.roundshot.com/guetsch/",
                "note": "360° panoramic cam — opens in new tab",
            },
            {
                "label": "Andermatt Village",
                "url": None,
                "link": "https://andermatt.roundshot.com/city/",
                "note": "360° panoramic cam — opens in new tab",
            },
        ],
        "color": "#1d4ed8",
    },
    {
        "name": "🏔️ Engelberg-Titlis",
        "subtitle": "Titlis Bergbahnen · Obwalden",
        "meta": "3,020 m top elevation · ~82 km pistes · SLF region: Central Switzerland",
        "coords": (46.8200, 8.4070),
        "webcam_page": "https://www.engelberg.ch/en/webcams/",
        "cams": [
            {
                "label": "Erlenwiese & Laub",
                "url": "https://www.engelberg.ch/uploads/tx_fmdwebcams/cam_99.jpg",
                "link": "https://www.engelberg.ch/en/webcams/",
            },
            {
                "label": "Rugghubelhütte",
                "url": "https://www.engelberg.ch/uploads/tx_fmdwebcams/cam_95.jpg",
                "link": "https://www.engelberg.ch/en/webcams/",
            },
            {
                "label": "Village",
                "url": "https://www.engelberg.ch/uploads/tx_fmdwebcams/cam_91.jpg",
                "link": "https://www.engelberg.ch/en/webcams/",
            },
        ],
        "color": "#059669",
    },
    {
        "name": "⛷️ Klewenalp",
        "subtitle": "Klewenalp-Stockhütte · Beckenried / Emmetten, Nidwalden",
        "meta": "1,611 m top elevation · family ski area · SLF region: Central Switzerland",
        "coords": (46.9401, 8.4732),
        "webcam_page": "https://www.klewenalp.ch/live-am-berg/webcams",
        "cams": [
            {
                "label": "Klewenalp / Ergglen",
                "url": "https://wtvpict.feratel.com/picture/37/4030.jpeg",
                "link": "https://www.klewenalp.ch/live-am-berg/webcams",
            },
            {
                "label": "Bergstation",
                "url": "https://wtvpict.feratel.com/picture/37/4031.jpeg",
                "link": "https://www.klewenalp.ch/live-am-berg/webcams",
            },
            {
                "label": "Stockhütte (Süd)",
                "url": "https://wtvpict.feratel.com/picture/37/4032.jpeg",
                "link": "https://www.klewenalp.ch/live-am-berg/webcams",
            },
        ],
        "color": "#7c3aed",
    },
]

with tab_webcams:
    st.subheader("📹 Live Webcams — Local Ski Areas")
    st.caption(
        f"Images refresh hourly · Last loaded: {datetime.now().strftime('%d %b %Y %H:%M')}"
        " · Click any link for the full live view"
    )

    # Map showing all three resorts
    resort_map_df = pd.DataFrame([
        {"name": r["name"], "lat": r["coords"][0], "lon": r["coords"][1]}
        for r in RESORTS
    ])
    resort_layer = pdk.Layer(
        "ScatterplotLayer",
        data=resort_map_df,
        get_position=["lon", "lat"],
        get_fill_color=[[29, 78, 216], [5, 150, 105], [124, 58, 237]],
        get_radius=4000,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=2,
    )
    resort_deck = pdk.Deck(
        layers=[resort_layer],
        initial_view_state=pdk.ViewState(latitude=46.85, longitude=8.45, zoom=9, pitch=20),
        tooltip={"text": "{name}"},
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    )
    st.pydeck_chart(resort_deck, use_container_width=True, height=280)

    st.markdown("---")

    # Cache-buster: update every hour so cached images don't stay stale
    cache_v = datetime.now().strftime("%Y%m%d%H")

    for resort in RESORTS:
        st.markdown(f"### {resort['name']}")
        st.markdown(
            f"<p class='resort-meta'>{resort['subtitle']}<br>{resort['meta']}</p>",
            unsafe_allow_html=True,
        )

        cols = st.columns(len(resort["cams"]))
        for col, cam in zip(cols, resort["cams"]):
            with col:
                if cam.get("url"):
                    # Display live image
                    st.markdown(
                        f'<a href="{cam["link"]}" target="_blank">'
                        f'<img src="{cam["url"]}?v={cache_v}" '
                        f'style="width:100%;border-radius:8px;border:1px solid #e5e7eb;" '
                        f'onerror="this.style.display=\'none\';this.nextSibling.style.display=\'block\'"/>'
                        f'</a>'
                        f'<div style="display:none;background:#f3f4f6;border-radius:8px;'
                        f'padding:40px 10px;text-align:center;color:#6b7280;font-size:0.8rem;">'
                        f'📷 Image unavailable<br><a href="{cam["link"]}" target="_blank">Open webcam →</a></div>'
                        f'<p class="webcam-label">{cam["label"]}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    # Roundshot / no direct image — show link card
                    st.markdown(
                        f'<a href="{cam["link"]}" target="_blank" style="text-decoration:none;">'
                        f'<div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;'
                        f'padding:30px 10px;text-align:center;cursor:pointer;">'
                        f'<div style="font-size:2rem;">🔭</div>'
                        f'<div style="font-size:0.82rem;color:#1d4ed8;margin-top:8px;font-weight:600;">'
                        f'Open 360° cam →</div>'
                        f'<div style="font-size:0.75rem;color:#6b7280;margin-top:4px;">{cam["label"]}</div>'
                        f'</div></a>'
                        f'<p class="webcam-label">{cam.get("note","")}</p>',
                        unsafe_allow_html=True,
                    )

        st.markdown(
            f"[🔗 All {resort['name'].split()[1]} webcams →]({resort['webcam_page']})",
        )
        st.markdown("---")

    # SLF Bulletin link
    st.info(
        "📋 **Current avalanche bulletin:** "
        "[slf.ch/en/avalanche-bulletin](https://www.slf.ch/en/avalanche-bulletin-and-snow-situation/)"
        " — updated daily at 08:00 and 17:00"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("Understanding the Avalanche Danger Scale")

    danger_info = [
        ("1", "Low",          "#61c050", "Generally safe conditions. Isolated triggering possible only on very steep slopes (>40°)."),
        ("2", "Moderate",     "#f5d63c", "Cautious route selection recommended. Triggering possible mainly on steep slopes (>35°)."),
        ("3", "Considerable", "#f47f21", "⚠️ Dangerous. Triggering possible on steep slopes (>30°). Natural avalanches possible."),
        ("4", "High",         "#e02020", "❗ Very dangerous. Triggering likely on many steep slopes. Natural avalanches expected."),
        ("5", "Very High",    "#1a0a0a", "🚫 Extreme. Travel in open terrain inadvisable. Massive natural avalanches expected."),
    ]

    for lvl, name, color, desc in danger_info:
        col_badge, col_text = st.columns([1, 7])
        col_badge.markdown(
            f'<div style="background:{color};color:{"#fff" if lvl not in ["2"] else "#333"};'
            f'border-radius:8px;padding:14px;text-align:center;font-size:1.6rem;font-weight:800;">'
            f'{lvl}</div>'
            f'<div style="text-align:center;font-weight:700;margin-top:4px;">{name}</div>',
            unsafe_allow_html=True,
        )
        col_text.markdown(f"**Level {lvl} — {name}**\n\n{desc}")
        st.markdown("")

    st.markdown("---")
    st.markdown("""
### About this dashboard

This dashboard scrapes live accident data from the
[WSL Institute for Snow and Avalanche Research SLF](https://www.slf.ch),
the official Swiss authority for avalanche warnings.

**How to read the map:**
- 🔵 Blue dot — accident with no burials
- 🟠 Orange dot — one or more people buried (survived)
- 🔴 Red dot — at least one fatality
- Dot size scales with the number of people caught

**Data source:** SLF accident reports, updated throughout the season.
Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) —
cite as *"Source: WSL SLF, retrieved [date]"*.

**Webcam images:** Engelberg webcams © engelberg.ch ·
Klewenalp webcams © feratel.com ·
Andermatt panoramas © roundshot.com
""")
