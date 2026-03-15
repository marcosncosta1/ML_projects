# 🏔️ Swiss Avalanche Risk Dashboard

Live interactive dashboard visualising avalanche accident data from Switzerland's official source —
the [WSL Institute for Snow and Avalanche Research SLF](https://www.slf.ch).

**[→ Live Demo](https://your-app.streamlit.app)** *(update link after deploying)*

---

## Features

| Tab | Description |
|-----|-------------|
| 🗺️ Map | Four pydeck views: scatter plot, heatmap, hexagon grid, 3D columns |
| 📊 Analytics | Weekly timeline, accidents by canton, activity breakdown, elevation histogram |
| 📹 Webcams | Live webcam feeds for Andermatt, Engelberg, and Klewenalp |
| ℹ️ About | Avalanche danger scale reference and data attribution |

## Stack

- **Streamlit** — UI and layout
- **pydeck** — interactive 3D map layers (CARTO tiles, no API key needed)
- **Plotly** — analytics charts
- **BeautifulSoup + requests** — live scraping from SLF
- **pyproj** — coordinate conversion (LV95 → WGS84)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo and branch → set **Main file path** to `app.py`
4. Click **Deploy** — your app will be live in ~2 minutes

## Data source

Accident data: [WSL SLF — All reported avalanche accidents in current season](https://www.slf.ch/en/avalanches/avalanches-and-avalanche-accidents/all-reported-avalanche-accidents-in-current-year/)
Licence: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — cite as *"Source: WSL SLF, retrieved [date]"*

Webcam images: © engelberg.ch · © feratel.com · © roundshot.com
