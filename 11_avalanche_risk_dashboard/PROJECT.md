# Avalanche Risk Dashboard

## Overview
Clean up and deploy the Swiss SLF scraper app as a polished full-stack web application with a proper landing page. Swiss-specific, visually compelling, and demonstrates full-stack engineering.

## Category
Software Engineering / Full-Stack / Data Visualization

## Stack
- **Python** — scraping backend (existing)
- **Streamlit or FastAPI + React** — frontend
- **Render or Railway** — deployment
- **Leaflet.js or Mapbox** — interactive map visualization
- **Scheduled job** (cron or Render cron) — daily data refresh

## Key Features
- Interactive map of Switzerland with avalanche danger levels per region
- Color-coded danger scale (1–5: Low to Very High)
- Historical trend charts per region
- Mobile-responsive design
- Auto-refreshes daily from SLF data source
- "About" section explaining avalanche danger scale for non-experts

## Architecture
```
SLF Website
    ↓ (scraper, daily cron)
Parsed Data Store (JSON / SQLite)
    ↓
FastAPI / Streamlit backend
    ↓
Frontend Map + Charts
    ↓
Deployed on Render/Railway
```

## Portfolio Value
- Swiss-specific = memorable and distinctive
- Full-stack: scraping + backend + frontend + deployment
- Real, live data makes it feel production-grade
- Shows initiative (built from a personal need or interest)

## Milestones
- [ ] Audit and clean up existing scraper code
- [ ] Proper data model + storage (SQLite or JSON files)
- [ ] Interactive map visualization
- [ ] Historical trend charts
- [ ] Landing page / about section
- [ ] Deploy on Render with daily cron refresh
- [ ] Custom domain (optional)
- [ ] README with screenshots

## Notes
<!-- Add implementation notes, decisions, and progress here -->
