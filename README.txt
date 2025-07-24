Energy Theft Detection Dashboard
Overview
This project is an Energy Theft Detection Dashboard developed as part of the Ikeja Electric (IE) Young Engineers Program. It uses machine learning to detect energy theft using simulated data for 7 streets, helping IE recover revenue and complement the Intelligent Data Box (IDB). The dashboard identifies high-risk theft cases and provides actionable insights for vigilance teams, with potential to scale to other Discos (e.g., Eko, Abuja).
Features

Machine Learning: Uses Scikit-learn Isolation Forest to flag high-risk buildings (flagged ≥3 days).
Key Metrics: 
Consumption score (1 - usage_kwh/expected_kwh, 20% weight).
Feeder score (45% weight, based on supplied vs. metered kWh ratio).
Payment history (15% weight, 0.0 for non-payers, 0.8–1.0 for payers).
Pattern deviation and location trust (combined 20% weight).


Visualizations: Heatmaps (building-level and street-ranking) and bar charts show theft probability and financial losses (₦209.5/kWh).
IDB Synergy: Complements IE’s tamper-proof IDB by prioritizing vigilance, reducing grid-wide deployment costs.
Simulated Data: Models 7 streets, ~100 buildings (Malls, Hotels, Offices, Apartments, Bungalows) with hourly data (Jul 1–14, 2025).
Outputs: Lists high-risk buildings for vigilance teams, exportable as CSV.

Pilot Results

Simulated Data: 7 streets, ~100 buildings, hourly data (Jul 1–14, 2025).
Savings: ₦1M in 10 days, ₦36.5M/year (simulated), scalable to ₦400M across IE’s 11 units.
Output: Identifies high-risk buildings (e.g., Mall_Street_1_1 flagged for 5 days) for vigilance action.

Installation

Clone the repository:git clone https://github.com/n0telvispresley/Energy-theft-detector


Install dependencies from requirements.txt:pip install -r requirements.txt


Run the Streamlit app:streamlit run energy_theft_detection_multihome.py



Dependencies

pandas==2.0.3
numpy==1.24.3
scikit-learn==1.2.2
streamlit==1.31.0
matplotlib==3.7.1
seaborn==0.12.2

Usage

The app uses built-in simulated data (7 streets, ~100 buildings, Jul 1–14, 2025).
Adjust weights (feeder score, consumption score, payment history, etc.) via sliders.
Filter by street, date, building type, and minimum theft probability.
View heatmaps, financial loss charts, and high-risk building summaries.
Export high-risk buildings as CSV for vigilance teams.

Future Enhancements

Integrate distribution transformer data for precise targeting (per IE lines manager suggestion).
Add CSV ingestion for real IE data (e.g., supplied_kwh, metered_kwh).
Explore software sales to other Discos (e.g., Eko, Abuja) for revenue generation.

Author
Elvis Ebenuwah, Summer Intern, Ikeja Electric, July 2025