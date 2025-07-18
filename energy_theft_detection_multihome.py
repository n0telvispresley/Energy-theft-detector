import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate data for 7 streets (2 weeks)
np.random.seed(42)
dates = pd.date_range(start="2025-07-01", end="2025-07-14", freq="h")
n_hours = len(dates)

# Define building types and consumption ranges (kWh/day)
building_types = {
    "Mall": (500, 1000, 1, 2),  # Min, max, min_count, max_count
    "Hotel": (200, 500, 1, 3),
    "Office": (100, 300, 2, 5),
    "Apartment": (20, 50, 2, 5),
    "Bungalow": (5, 15, 3, 10)
}

# Simulate 7 streets with 10-25 buildings
streets = [f"Street_{i}" for i in range(1, 8)]
street_buildings = {s: np.random.randint(10, 26) for s in streets}
buildings = []
for street in streets:
    n_buildings = street_buildings[street]
    type_counts = {t: np.random.randint(c[2], c[3] + 1) for t, c in building_types.items()}
    total_types = sum(type_counts.values())
    if total_types > n_buildings:
        type_counts = {t: int(c * n_buildings / total_types) for t, c in type_counts.items()}
        total_types = sum(type_counts.values())
    while total_types < n_buildings:
        type_counts["Bungalow"] += 1
        total_types += 1
    for btype, count in type_counts.items():
        for i in range(1, count + 1):
            buildings.append((f"{btype}_{street}_{i}", btype, street))

# Simulate hourly data
data = []
for bid, btype, street in buildings:
    low, high = building_types[btype][:2]
    daily_avg = np.random.uniform(low, high)
    hourly_base = daily_avg / 24
    usage = [max(0, hourly_base * np.random.uniform(0.9, 1.1)) for _ in range(n_hours)]
    payment_history = 0.0 if bid not in [f"Mall_{s}_1" for s in streets] + [f"Bungalow_{s}_1" for s in streets] else np.random.uniform(0.8, 1.0)
    location_trust = 0.3 if street not in ["Street_1", "Street_2"] else 0.7
    phase_current = np.random.normal(daily_avg / 24 * 10, 1.5, n_hours)
    data.append(pd.DataFrame({
        "timestamp": dates,
        "street_id": street,
        "building_id": bid,
        "building_type": btype,
        "phase_current": phase_current,
        "neutral_current": phase_current.copy(),
        "usage_kwh": usage,
        "payment_history": payment_history,
        "location_trust": location_trust
    }))

data = pd.concat(data, ignore_index=True)

# Simulate persistent theft
theft_buildings = {f"Mall_{s}_1": (pd.to_datetime("2025-07-03"), pd.to_datetime("2025-07-07")) for s in streets if f"Mall_{s}_1" in [b[0] for b in buildings]}
theft_buildings.update({f"Bungalow_{s}_1": (pd.to_datetime("2025-07-02"), pd.to_datetime("2025-07-08")) for s in streets if f"Bungalow_{s}_1" in [b[0] for b in buildings]})
for bid, (start, end) in theft_buildings.items():
    mask = (data["building_id"] == bid) & (data["timestamp"] >= start) & (data["timestamp"] <= end)
    data.loc[mask, "usage_kwh"] *= np.random.uniform(0.3, 0.5)
    data.loc[mask, "neutral_current"] *= 0.5

# Calculate feeder data
data["date"] = data["timestamp"].dt.date
daily_data = data.groupby(["date", "street_id", "building_id", "building_type"]).agg({
    "usage_kwh": "sum",
    "payment_history": "mean",
    "location_trust": "mean"
}).reset_index()
daily_data["expected_kwh"] = daily_data.groupby("building_id")["usage_kwh"].transform("mean")

# Simulate supplied kWh independently
feeder_data = daily_data.groupby(["date", "street_id"]).agg({"building_id": "count", "building_type": lambda x: list(x)}).reset_index()
feeder_data["max_expected_kwh"] = feeder_data.apply(
    lambda row: sum(building_types[btype][1] for btype in row["building_type"]), axis=1
)
feeder_data["supplied_kwh"] = feeder_data["max_expected_kwh"] * np.random.uniform(1.1, 1.2)
feeder_data["metered_kwh"] = daily_data.groupby(["date", "street_id"])["usage_kwh"].sum().reset_index()["usage_kwh"]
feeder_data["feeder_ratio"] = feeder_data["supplied_kwh"] / feeder_data["metered_kwh"]
feeder_data["feeder_score"] = ((feeder_data["feeder_ratio"] - 1.1) / (2 - 1.1)).clip(0, 1)
feeder_data["financial_loss_naira"] = (feeder_data["supplied_kwh"] - feeder_data["metered_kwh"]) * 209.5

# Merge feeder data
daily_data = daily_data.merge(feeder_data[["date", "street_id", "feeder_score"]], on=["date", "street_id"])

# Streamlit dashboard
st.title("IKEDC Energy Theft Detection Dashboard")
st.write("Detect high-risk buildings (flagged ≥3 days), prioritize streets, and estimate savings (₦209.5/kWh)")

# Weight adjustment sliders
st.subheader("Adjust Weights for Theft Probability")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    feeder_weight = st.slider("Feeder Score Weight", 0.0, 1.0, 0.45, 0.05)
with col2:
    consumption_weight = st.slider("Consumption Score Weight", 0.0, 1.0, 0.2, 0.05)
with col3:
    payment_weight = st.slider("Payment History Weight", 0.0, 1.0, 0.15, 0.05)
with col4:
    deviation_weight = st.slider("Pattern Deviation Weight", 0.0, 1.0, 0.15, 0.05)
with col5:
    location_weight = st.slider("Location Trust Weight", 0.0, 1.0, 0.05, 0.05)

# Normalize weights to sum to 1
total_weight = feeder_weight + consumption_weight + payment_weight + deviation_weight + location_weight
if total_weight > 0:
    feeder_weight /= total_weight
    consumption_weight /= total_weight
    payment_weight /= total_weight
    deviation_weight /= total_weight
    location_weight /= total_weight
else:
    st.error("Total weight cannot be zero. Using default weights.")
    feeder_weight, consumption_weight, payment_weight, deviation_weight, location_weight = 0.45, 0.2, 0.15, 0.15, 0.05

# Calculate weighted features
daily_data["consumption_score"] = (1 - daily_data["usage_kwh"] / daily_data["expected_kwh"]).clip(0, 1)
daily_data["pattern_deviation"] = abs(daily_data["usage_kwh"] - daily_data["expected_kwh"]) / daily_data["expected_kwh"]
daily_data["theft_probability"] = (
    feeder_weight * daily_data["feeder_score"] +
    consumption_weight * daily_data["consumption_score"] +
    payment_weight * daily_data["payment_history"] +
    deviation_weight * daily_data["pattern_deviation"] +
    location_weight * daily_data["location_trust"]
).clip(0, 1)

# Train Scikit-learn Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
features = daily_data[["theft_probability", "feeder_score", "consumption_score"]]
daily_data["is_theft"] = model.fit_predict(features) == -1

# Filter by street, date, building type, and probability
st.subheader("Filter Data for Building-Level Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    street = st.selectbox("Select Street", streets)
with col2:
    date = st.date_input("Select Date", value=pd.to_datetime("2025-07-03"))
with col3:
    building_type = st.selectbox("Building Type", ["All"] + list(building_types.keys()))
min_prob = st.slider("Minimum Theft Probability", 0.0, 1.0, 0.0)
filtered_data = daily_data[(daily_data["street_id"] == street) & (daily_data["date"] == date)]
if building_type != "All":
    filtered_data = filtered_data[filtered_data["building_type"] == building_type]
filtered_data = filtered_data[filtered_data["theft_probability"] >= min_prob]

# Display filtered data
st.subheader(f"Daily Theft Probability for {street} (Filtered)")
st.dataframe(filtered_data[["building_id", "building_type", "usage_kwh", "theft_probability", "is_theft"]].round(3))

# Savings estimates
st.subheader("Potential Savings Estimates (₦209.5/kWh)")
daily_savings = feeder_data[feeder_data["date"] == date]["financial_loss_naira"].sum()
total_savings = feeder_data["financial_loss_naira"].sum()
avg_daily_savings = total_savings / 14  # 14 days
monthly_savings = avg_daily_savings * 30
yearly_savings = avg_daily_savings * 365
st.write(f"Potential Savings for {date}: ₦{daily_savings:,.2f}")
st.write(f"Total Savings for Period (Jul 1–14, 2025): ₦{total_savings:,.2f}")
st.write(f"Estimated Monthly Savings (30 days): ₦{monthly_savings:,.2f}")
st.write(f"Estimated Yearly Savings (365 days): ₦{yearly_savings:,.2f}")

# Building-level heatmap
st.subheader(f"Theft Probability Heatmap for {street} (White to Red)")
pivot_data = daily_data[daily_data["street_id"] == street]
if building_type != "All":
    pivot_data = pivot_data[pivot_data["building_type"] == building_type]
pivot_data = pivot_data.pivot(index="building_id", columns="date", values="theft_probability")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(pivot_data, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Theft Probability"})
ax.set_xlabel("Date")
ax.set_ylabel("Building ID")
ax.set_title(f"Theft Probability for {street} (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Street-ranking heatmap
st.subheader("Street-Ranking Heatmap for Theft Probability or Technical Losses (White to Red)")
street_pivot = daily_data.groupby(["date", "street_id"])["theft_probability"].mean().reset_index()
street_pivot = street_pivot.pivot(index="street_id", columns="date", values="theft_probability")
street_order = street_pivot.mean(axis=1).sort_values(ascending=False).index
street_pivot = street_pivot.loc[street_order]
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(street_pivot, cmap="YlOrRd", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Average Theft Probability"})
ax.set_xlabel("Date")
ax.set_ylabel("Street ID")
ax.set_title("Street-Ranking Heatmap for Theft Probability or Technical Losses (kWh)")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Financial loss bar chart
st.subheader("Estimated Daily Financial Loss per Street (₦)")
loss_data = feeder_data[feeder_data["date"] == date].set_index("street_id")["financial_loss_naira"]
loss_data = loss_data.reindex(street_order, fill_value=0)
fig, ax = plt.subplots(figsize=(10, 6))
loss_data.plot(kind="bar", color=["#FF6B6B" if i < 2 else "#FFADAD" for i in range(len(loss_data))], ax=ax)
ax.set_xlabel("Street ID")
ax.set_ylabel("Loss (₦)")
ax.set_title(f"Estimated Daily Financial Loss on {date} (₦209.5/kWh)")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Summary of high-risk buildings (flagged ≥3 days)
st.subheader("High-Risk Buildings Summary (Flagged ≥3 Days)")
theft_count = daily_data[daily_data["is_theft"]].groupby(["street_id", "building_id"]).size().reset_index(name="days_flagged")
high_risk = theft_count[theft_count["days_flagged"] >= 3]
st.write(f"Total High-Risk Buildings (Flagged ≥3 Days): {len(high_risk)}")
if len(high_risk) > 0:
    for _, row in high_risk.iterrows():
        st.write(f"{row['street_id']} - {row['building_id']}: Flagged for {row['days_flagged']} days")
else:
    st.write("No buildings flagged for ≥3 days.")