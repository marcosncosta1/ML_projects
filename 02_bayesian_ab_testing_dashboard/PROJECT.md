# Bayesian A/B Testing Dashboard

## Overview
An interactive Streamlit app where users input conversion data (visitors, conversions per variant) and receive a full Bayesian posterior analysis — credible intervals, probability of superiority, and expected lift.

## Category
Machine Learning / Statistics / Business Analytics

## Stack
- **PyMC** — Bayesian inference (Beta-Binomial model)
- **ArviZ** — posterior visualization
- **Streamlit** — UI with input widgets and charts
- **Matplotlib / Plotly** — posterior distribution plots

## Key Features
- Input: number of visitors + conversions for variant A and B
- Output:
  - Posterior distributions for each variant's conversion rate
  - Probability that B > A
  - Credible intervals (e.g. 94% HDI)
  - Expected lift with uncertainty
  - Bayesian decision recommendation
- Option to simulate data for demo mode

## Architecture
1. User inputs raw counts via Streamlit sliders/text inputs
2. PyMC samples Beta posterior for each variant
3. ArviZ/Plotly renders posterior plots
4. Summary statistics computed from posterior samples

## Portfolio Value
- Demonstrates Bayesian ML in a business-relevant, intuitive context
- Directly applicable to product/growth roles
- Differentiator vs. candidates who only know frequentist statistics

## Milestones
- [ ] Beta-Binomial model in PyMC
- [ ] Streamlit layout with inputs + charts
- [ ] Demo mode with synthetic data
- [ ] Deploy on Streamlit Cloud
- [ ] README with explanation of Bayesian vs frequentist framing

## Notes
<!-- Add implementation notes, decisions, and progress here -->
