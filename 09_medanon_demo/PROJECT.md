# MedAnon Demo

## Overview
Package the existing MedAnon project with a polished Streamlit UI, sample anonymized PDFs, and a proper README with an architecture diagram. This project is already built — the focus is on presentation quality.

## Category
Software Engineering / NLP / Healthcare

## Stack
- **Streamlit** — polished UI
- **Existing MedAnon backend** — NER-based anonymization pipeline
- **Sample PDFs** — pre-loaded examples for live demo without user data
- **Draw.io / Excalidraw** — architecture diagram

## Presentation Goals
- Clean Streamlit UI with:
  - File upload (PDF or text)
  - Side-by-side: original vs. anonymized output
  - Highlighted redacted entities (color-coded by type: name, date, location, etc.)
  - Download anonymized output
- Sample anonymized PDFs in `/examples/` for instant demo
- Architecture diagram showing the anonymization pipeline
- Clear README with: motivation, usage, limitations, privacy guarantees

## Repository Structure
```
medanon/
├── app.py               # Streamlit app
├── medanon/             # Core anonymization library
├── examples/            # Sample input + output files
├── docs/
│   └── architecture.png
├── README.md
└── requirements.txt
```

## Portfolio Value
- Already built — just needs polish
- Healthcare + NLP intersection is highly valuable
- Privacy/anonymization is a growing compliance concern

## Milestones
- [ ] Audit existing code for clean-up
- [ ] Build Streamlit UI with side-by-side view
- [ ] Create sample anonymized PDF examples
- [ ] Draw architecture diagram
- [ ] Write README (motivation, usage, limitations)
- [ ] Deploy on Streamlit Cloud

## Notes
<!-- Add implementation notes, decisions, and progress here -->
