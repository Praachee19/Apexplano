import json
import math
import textwrap
from io import BytesIO
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Apex Footwear Wall Planogram Agent", layout="wide")

# ============================================================
# APEX FOOTWEAR WALL PLANOGRAM AGENT
# Brand context: apex4u.com style footwear assortment
# Wall: 15 ft wide x 8 ft high
# Merchandise height: 6 ft only. Top 2 ft reserved for branding/signage.
# Features:
# - Synthetic one-year sales data
# - Upload template
# - Sales per sq ft, GMROI, inventory turns, weeks cover
# - Out-of-stock and out-of-display control
# - Store customized planogram
# - Explainable AI table
# - Optional Ollama local explanation
# - Downloadable recommendation CSV and planogram PNG
# ============================================================

WALL_WIDTH_FT = 15
WALL_HEIGHT_FT = 8
MERCH_HEIGHT_FT = 6
BAY_WIDTH_FT = 3
BAY_COUNT = int(WALL_WIDTH_FT / BAY_WIDTH_FT)
SHELF_LEVELS = ["Bottom", "Lower", "Mid", "Upper Mid", "Eye Level", "Top Merch"]
SHELF_Y_IN = {"Bottom": 6, "Lower": 18, "Mid": 30, "Upper Mid": 42, "Eye Level": 54, "Top Merch": 66}

GENDERS = ["Men", "Women", "Kids"]
CATEGORIES = [
    "Formal Shoes", "Casual Shoes", "Sneakers", "Sandals", "Slippers", "School Shoes", "Bags"
]
PRODUCT_LINES = ["Apex", "Venturini", "Maverick", "Sprint", "Moochie", "Nino Rossi", "Twinkler", "Ipanema"]
COLOURS = ["Black", "Brown", "Tan", "Navy", "White", "Beige", "Red", "Pink", "Blue", "Grey"]
SIZES = ["35", "36", "37", "38", "39", "40", "41", "42", "43", "44"]
STORES = ["Dhaka Flagship", "Gulshan Store", "Chattogram Store", "Online Store"]

CATEGORY_COLORS = {
    "Formal Shoes": "#3b3b3b",
    "Casual Shoes": "#8d6e63",
    "Sneakers": "#607d8b",
    "Sandals": "#d95f02",
    "Slippers": "#7570b3",
    "School Shoes": "#1b9e77",
    "Bags": "#e7298a",
}

BENCHMARKS = {
    "sales_per_sqft_monthly": 4500.0,
    "gmroi_good": 2.5,
    "inventory_turns_good_annual": 4.0,
    "weeks_cover_target": 8.0,
    "weeks_cover_high": 18.0,
    "display_fill_target": 0.90,
    "min_facings_per_sku": 1,
    "max_facings_per_sku": 6,
}

UPLOAD_COLUMNS = [
    "date", "store", "product_line", "gender", "category", "style", "colour", "size",
    "sku", "opening_stock_units", "receipts_units", "sales_units", "closing_stock_units",
    "display_capacity_units", "display_stock_units", "net_sales_value", "full_price",
    "avg_realized_price", "cost_per_unit", "gross_margin_pct", "markdown_pct",
    "age_days", "season_type", "last_year_sales_units"
]


def create_upload_template() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": "2026-03-01",
            "store": "Dhaka Flagship",
            "product_line": "Apex",
            "gender": "Men",
            "category": "Formal Shoes",
            "style": "Men Formal Slip On",
            "colour": "Black",
            "size": "42",
            "sku": "APX-MEN-FORMAL-BLK-42",
            "opening_stock_units": 20,
            "receipts_units": 5,
            "sales_units": 6,
            "closing_stock_units": 19,
            "display_capacity_units": 3,
            "display_stock_units": 2,
            "net_sales_value": 20340,
            "full_price": 3990,
            "avg_realized_price": 3390,
            "cost_per_unit": 1700,
            "gross_margin_pct": 0.50,
            "markdown_pct": 0.15,
            "age_days": 55,
            "season_type": "Core",
            "last_year_sales_units": 5,
        }
    ], columns=UPLOAD_COLUMNS)


def generate_synthetic_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-04-06", periods=52, freq="W-SUN")
    rows = []

    base_price = {
        "Formal Shoes": 3990, "Casual Shoes": 2990, "Sneakers": 3490,
        "Sandals": 1990, "Slippers": 1290, "School Shoes": 2490, "Bags": 2990,
    }
    category_velocity = {
        "Formal Shoes": 0.9, "Casual Shoes": 1.0, "Sneakers": 1.15,
        "Sandals": 1.35, "Slippers": 1.25, "School Shoes": 1.10, "Bags": 0.55,
    }
    gender_factor = {"Men": 1.15, "Women": 1.05, "Kids": 0.85}
    store_factor = {"Dhaka Flagship": 1.25, "Gulshan Store": 1.15, "Chattogram Store": 0.95, "Online Store": 1.05}

    for store in STORES:
        for product_line in PRODUCT_LINES:
            line_factor = rng.uniform(0.75, 1.25)
            for gender in GENDERS:
                allowed_categories = CATEGORIES.copy()
                if gender == "Kids":
                    allowed_categories = ["School Shoes", "Sandals", "Slippers", "Sneakers"]
                for category in allowed_categories:
                    for colour in rng.choice(COLOURS, size=4, replace=False):
                        size_pool = SIZES if gender != "Kids" else ["35", "36", "37", "38", "39", "40"]
                        for size in rng.choice(size_pool, size=4, replace=False):
                            style = f"{gender} {category}"
                            sku = f"{product_line[:3].upper()}-{gender[:1]}-{category[:3].upper()}-{colour[:3].upper()}-{size}"
                            price = base_price[category] * rng.uniform(0.85, 1.25)
                            cost = price * rng.uniform(0.38, 0.55)
                            margin_pct = 1 - (cost / price)
                            season_type = rng.choice(["Core", "Seasonal", "New Arrival", "Clearance Risk"], p=[0.45, 0.25, 0.20, 0.10])
                            stock = int(rng.integers(6, 32))
                            age = int(rng.integers(15, 220))
                            ly_sales = max(0, int(3.5 * category_velocity[category] * gender_factor[gender] * store_factor[store] * line_factor + rng.normal(0, 1)))

                            for dt in dates:
                                eid_boost = 1.45 if dt.month in [3, 4, 5, 6] else 1.0
                                winter_formal_boost = 1.15 if category in ["Formal Shoes", "Casual Shoes"] and dt.month in [11, 12, 1] else 1.0
                                markdown_pct = float(rng.choice([0, 0.10, 0.15, 0.25, 0.35, 0.50], p=[0.45, 0.18, 0.14, 0.12, 0.08, 0.03]))
                                demand = 2.8 * category_velocity[category] * gender_factor[gender] * store_factor[store] * line_factor * eid_boost * winter_formal_boost
                                demand *= (1 + markdown_pct * 1.1)
                                demand *= max(0.45, 1 - age / 520)
                                units = int(min(stock, rng.poisson(max(0.2, demand))))
                                receipts = int(rng.integers(0, 10) if stock < 8 else rng.integers(0, 3))
                                opening = stock
                                closing = max(0, opening + receipts - units)
                                realized_price = price * (1 - markdown_pct)
                                net_sales = units * realized_price
                                display_capacity = 2 if category != "Bags" else 1
                                display_stock = min(display_capacity, closing, int(rng.integers(0, display_capacity + 1)))

                                rows.append({
                                    "date": dt,
                                    "store": store,
                                    "product_line": product_line,
                                    "gender": gender,
                                    "category": category,
                                    "style": style,
                                    "colour": colour,
                                    "size": size,
                                    "sku": sku,
                                    "opening_stock_units": opening,
                                    "receipts_units": receipts,
                                    "sales_units": units,
                                    "closing_stock_units": closing,
                                    "display_capacity_units": display_capacity,
                                    "display_stock_units": display_stock,
                                    "net_sales_value": round(net_sales, 2),
                                    "full_price": round(price, 2),
                                    "avg_realized_price": round(realized_price, 2),
                                    "cost_per_unit": round(cost, 2),
                                    "gross_margin_pct": round(margin_pct, 2),
                                    "markdown_pct": markdown_pct,
                                    "age_days": age,
                                    "season_type": season_type,
                                    "last_year_sales_units": ly_sales,
                                })
                                stock = closing
                                age += 7
    return pd.DataFrame(rows)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for col in UPLOAD_COLUMNS:
        if col not in out.columns:
            out[col] = 0
    numeric = [c for c in UPLOAD_COLUMNS if c not in ["date", "store", "product_line", "gender", "category", "style", "colour", "size", "sku", "season_type"]]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["gross_margin_value"] = out["net_sales_value"] - (out["sales_units"] * out["cost_per_unit"])
    out["avg_inventory_units"] = (out["opening_stock_units"] + out["closing_stock_units"]) / 2
    out["avg_inventory_cost"] = out["avg_inventory_units"] * out["cost_per_unit"]
    out["gmroi"] = np.where(out["avg_inventory_cost"] > 0, out["gross_margin_value"] / out["avg_inventory_cost"], 0)
    out["inventory_turnover"] = np.where(out["avg_inventory_units"] > 0, out["sales_units"] / out["avg_inventory_units"], 0)
    out["weeks_cover"] = np.where(out["sales_units"] > 0, out["closing_stock_units"] / out["sales_units"], 999)
    out["sell_through"] = np.where(out["opening_stock_units"] + out["receipts_units"] > 0, out["sales_units"] / (out["opening_stock_units"] + out["receipts_units"]), 0)
    out["oos_flag"] = (out["closing_stock_units"] <= 0).astype(int)
    out["ood_flag"] = (out["display_stock_units"] <= 0).astype(int)
    return out


def forecast_and_recommend(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["store", "product_line", "gender", "category", "style", "colour", "size", "sku", "season_type"]
    rows = []
    for _, g in df.sort_values("date").groupby(group_cols):
        recent = g.tail(8)
        if recent.empty:
            continue
        weights = np.arange(1, len(recent) + 1)
        recent_sales = float(np.average(recent["sales_units"], weights=weights))
        ly = float(recent["last_year_sales_units"].tail(4).mean())
        markdown = float(recent["markdown_pct"].tail(4).mean())
        latest = recent.iloc[-1]
        season_mult = {"Core": 1.08, "Seasonal": 1.00, "New Arrival": 1.15, "Clearance Risk": 0.72}.get(latest["season_type"], 1.0)
        forecast_weekly = max(0.05, ((0.65 * recent_sales) + (0.35 * ly)) * (1 + markdown * 0.7) * season_mult)
        forecast_4w = forecast_weekly * 4
        rows.append({
            **{c: latest[c] for c in group_cols},
            "forecast_weekly_units": round(forecast_weekly, 2),
            "forecast_4w_units": round(forecast_4w, 2),
            "latest_stock_units": latest["closing_stock_units"],
            "latest_display_stock_units": latest["display_stock_units"],
            "latest_display_capacity_units": latest["display_capacity_units"],
            "latest_weeks_cover": latest["weeks_cover"],
            "latest_sell_through": latest["sell_through"],
            "latest_gmroi": latest["gmroi"],
            "latest_inventory_turnover": latest["inventory_turnover"],
            "latest_age_days": latest["age_days"],
            "latest_markdown_pct": latest["markdown_pct"],
            "latest_full_price": latest["full_price"],
            "latest_cost_per_unit": latest["cost_per_unit"],
            "last_year_sales_units": latest["last_year_sales_units"],
        })
    rec = pd.DataFrame(rows)
    if rec.empty:
        return rec

    rec["display_gap_units"] = np.maximum(0, rec["latest_display_capacity_units"] - rec["latest_display_stock_units"])
    rec["replenish_units"] = np.maximum(0, np.ceil((rec["forecast_weekly_units"] * 2.0) - rec["latest_stock_units"]))
    rec["score_velocity"] = rec["forecast_weekly_units"] * 18
    rec["score_margin"] = rec["latest_gmroi"].clip(lower=0, upper=6) * 12
    rec["score_stockout"] = np.where(rec["latest_stock_units"] <= rec["forecast_weekly_units"], 25, 0)
    rec["score_display_gap"] = np.where(rec["display_gap_units"] > 0, 18, 0)
    rec["score_freshness"] = np.where(rec["season_type"] == "New Arrival", 12, np.where(rec["latest_age_days"] > 180, -15, 0))
    rec["score_overstock_penalty"] = np.where(rec["latest_weeks_cover"] > BENCHMARKS["weeks_cover_high"], -25, 0)
    rec["priority_score"] = (rec[[c for c in rec.columns if c.startswith("score_")]].sum(axis=1)).clip(lower=1)

    actions, levels, reasons, facings = [], [], [], []
    for _, r in rec.iterrows():
        if r["latest_stock_units"] <= 0 or r["display_gap_units"] > 0:
            action = "Replenish display now"
            reason = "Stock or display is below minimum. This prevents out-of-stock and out-of-display leakage."
        elif r["latest_weeks_cover"] > BENCHMARKS["weeks_cover_high"] and r["forecast_4w_units"] < 4:
            action = "Move down wall or mark down"
            reason = "High stock cover and weak demand. Do not give prime wall space."
        elif r["latest_gmroi"] >= BENCHMARKS["gmroi_good"] and r["forecast_weekly_units"] >= 2.5:
            action = "Prime wall placement"
            reason = "Strong demand and GMROI. Give better visibility and more facings."
        elif r["season_type"] == "New Arrival":
            action = "Feature as newness"
            reason = "New arrivals need visibility to create trial and style discovery."
        else:
            action = "Maintain standard placement"
            reason = "Balanced sales, stock and margin position."

        if action in ["Prime wall placement", "Feature as newness"]:
            shelf = "Eye Level" if r["category"] in ["Sandals", "Casual Shoes", "Sneakers"] else "Upper Mid"
        elif action == "Move down wall or mark down":
            shelf = "Bottom"
        elif r["category"] in ["Bags"]:
            shelf = "Top Merch"
        elif r["category"] in ["Slippers", "School Shoes"]:
            shelf = "Lower"
        else:
            shelf = "Mid"

        facing = int(np.clip(math.ceil(r["priority_score"] / 35), BENCHMARKS["min_facings_per_sku"], BENCHMARKS["max_facings_per_sku"]))
        actions.append(action)
        levels.append(shelf)
        reasons.append(reason)
        facings.append(facing)

    rec["recommended_action"] = actions
    rec["recommended_shelf_level"] = levels
    rec["xai_reason"] = reasons
    rec["recommended_facings"] = facings
    return rec.sort_values("priority_score", ascending=False)


def compute_kpis(df: pd.DataFrame, rec: pd.DataFrame, wall_sqft: float) -> Dict[str, float]:
    recent = df.sort_values("date").groupby("sku").tail(12)
    monthly_sales = recent["net_sales_value"].sum() / 3
    gm = recent["gross_margin_value"].sum()
    inv_cost = recent["avg_inventory_cost"].mean()
    avg_inv = recent["avg_inventory_units"].mean()
    units = recent["sales_units"].sum()
    return {
        "monthly_sales": float(monthly_sales),
        "sales_per_sqft": float(monthly_sales / max(wall_sqft, 1)),
        "gmroi": float(gm / max(inv_cost, 1)),
        "annualized_inventory_turns": float((units / max(avg_inv, 1)) * (52 / 12)),
        "oos_skus": int((rec["latest_stock_units"] <= 0).sum()) if not rec.empty else 0,
        "ood_skus": int((rec["latest_display_stock_units"] <= 0).sum()) if not rec.empty else 0,
        "prime_wall_skus": int((rec["recommended_action"] == "Prime wall placement").sum()) if not rec.empty else 0,
        "markdown_or_move_down": int((rec["recommended_action"] == "Move down wall or mark down").sum()) if not rec.empty else 0,
    }


def make_planogram(rec: pd.DataFrame) -> pd.DataFrame:
    if rec.empty:
        return rec
    width_in = WALL_WIDTH_FT * 12
    bay_width_in = BAY_WIDTH_FT * 12
    placement_rows = []
    shelves = []
    for bay in range(1, BAY_COUNT + 1):
        for level in SHELF_LEVELS:
            shelves.append({"bay": bay, "level": level, "used_in": 0})

    draw_width_by_category = {"Bags": 16, "Formal Shoes": 13, "Casual Shoes": 13, "Sneakers": 14, "Sandals": 12, "Slippers": 11, "School Shoes": 12}
    level_rank = {lvl: i for i, lvl in enumerate(["Eye Level", "Upper Mid", "Mid", "Lower", "Top Merch", "Bottom"])}
    work = rec.sort_values(by=["recommended_shelf_level", "priority_score"], key=lambda s: s.map(level_rank).fillna(99) if s.name == "recommended_shelf_level" else s, ascending=[True, False])

    for _, row in work.iterrows():
        facing_left = int(row["recommended_facings"])
        item_width = draw_width_by_category.get(row["category"], 12)
        while facing_left > 0:
            target_idx = None
            for i, shelf in enumerate(shelves):
                if shelf["level"] == row["recommended_shelf_level"] and shelf["used_in"] + item_width <= bay_width_in:
                    target_idx = i
                    break
            if target_idx is None:
                for i, shelf in enumerate(shelves):
                    if shelf["used_in"] + item_width <= bay_width_in:
                        target_idx = i
                        break
            if target_idx is None:
                break
            shelf = shelves[target_idx]
            placement_rows.append({
                "bay": shelf["bay"],
                "shelf_level": shelf["level"],
                "x_in": shelf["used_in"],
                "width_in": item_width,
                "sku": row["sku"],
                "product_line": row["product_line"],
                "gender": row["gender"],
                "category": row["category"],
                "colour": row["colour"],
                "size": row["size"],
                "action": row["recommended_action"],
            })
            shelves[target_idx]["used_in"] += item_width
            facing_left -= 1
    return pd.DataFrame(placement_rows)


def draw_planogram(plan: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(16, 8))
    wall_width_in = WALL_WIDTH_FT * 12
    wall_height_in = WALL_HEIGHT_FT * 12
    merch_height_in = MERCH_HEIGHT_FT * 12
    bay_width_in = BAY_WIDTH_FT * 12

    ax.add_patch(patches.Rectangle((0, 0), wall_width_in, wall_height_in, fill=False, linewidth=2, edgecolor="black"))
    ax.add_patch(patches.Rectangle((0, merch_height_in), wall_width_in, wall_height_in - merch_height_in, facecolor="#f7f7f7", edgecolor="black", linewidth=1))
    ax.text(wall_width_in / 2, merch_height_in + 12, "APEX BRAND HEADER / CAMPAIGN AREA. No merchandising above 6 ft", ha="center", va="center", fontsize=11, fontweight="bold")

    for bay in range(1, BAY_COUNT + 1):
        x0 = (bay - 1) * bay_width_in
        ax.add_line(plt.Line2D([x0, x0], [0, wall_height_in], linewidth=1, color="grey"))
        ax.text(x0 + bay_width_in / 2, 2, f"Bay {bay}", ha="center", fontsize=9)
    ax.add_line(plt.Line2D([wall_width_in, wall_width_in], [0, wall_height_in], linewidth=1, color="grey"))

    for level, y in SHELF_Y_IN.items():
        ax.add_line(plt.Line2D([0, wall_width_in], [y, y], linewidth=1.6, color="black"))
        ax.text(-8, y + 2, level, ha="right", fontsize=8)

    if not plan.empty:
        for _, row in plan.iterrows():
            x0 = (row["bay"] - 1) * bay_width_in + row["x_in"]
            y0 = SHELF_Y_IN[row["shelf_level"]]
            color = CATEGORY_COLORS.get(row["category"], "#cccccc")
            ax.add_patch(patches.Rectangle((x0, y0), row["width_in"], 8, facecolor=color, edgecolor="black", linewidth=0.8))
            ax.text(x0 + row["width_in"] / 2, y0 + 4, f"{row['category'][:4]}\n{row['gender'][0]} {row['size']}", ha="center", va="center", fontsize=5, color="white")

    ax.set_xlim(-20, wall_width_in + 12)
    ax.set_ylim(0, wall_height_in + 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Apex Footwear Wall Planogram. 15 ft x 8 ft wall, merchandising only till 6 ft", fontsize=13, fontweight="bold")
    handles = [patches.Patch(facecolor=v, edgecolor="black", label=k) for k, v in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def ollama_available(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def ask_ollama(base_url: str, model_name: str, kpis: Dict[str, float], rec: pd.DataFrame) -> str:
    sample = rec.head(15)[["sku", "product_line", "gender", "category", "forecast_4w_units", "latest_stock_units", "latest_display_stock_units", "latest_gmroi", "latest_weeks_cover", "recommended_action", "recommended_shelf_level", "xai_reason"]].to_dict(orient="records")
    prompt = f"""
You are a footwear retail space planning expert.
Explain this Apex footwear wall planogram in plain English.
Focus on why product categories are placed on each shelf, which SKUs need display replenishment, which SKUs should move down or be marked down, and how this plan reduces lost sales.
Keep it under 300 words.

KPIs:
{json.dumps(kpis, indent=2)}

Top recommendations:
{json.dumps(sample, indent=2)}
""".strip()
    try:
        r = requests.post(f"{base_url}/api/generate", json={"model": model_name, "prompt": prompt, "stream": False}, timeout=45)
        r.raise_for_status()
        return r.json().get("response", "No response from Ollama.")
    except Exception as e:
        return f"Ollama call failed. {e}"


# -----------------------------
# UI
# -----------------------------
st.title("Apex Footwear Planogram Agent")
st.caption("Store-specific wall merchandising planner for Apex footwear. Synthetic one-year sales included. Upload-ready template included.")

with st.sidebar:
    st.header("Planner Controls")
    data_mode = st.radio("Data source", ["Synthetic Apex demo", "Upload CSV"], index=0)
    store_filter = st.selectbox("Store", ["All"] + STORES, index=1)
    gender_filter = st.selectbox("Gender", ["All"] + GENDERS, index=0)
    category_filter = st.selectbox("Category", ["All"] + CATEGORIES, index=0)
    benchmark_sales_sqft = st.number_input("Benchmark sales / sq ft / month", value=BENCHMARKS["sales_per_sqft_monthly"], step=100.0)
    use_ollama = st.checkbox("Use Ollama local explanation", value=False)
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    ollama_model = st.text_input("Ollama model", value="llama3.1:8b")

st.markdown("### Data upload template")
template = create_upload_template()
st.dataframe(template, use_container_width=True)
st.download_button("Download upload template CSV", data=template.to_csv(index=False).encode("utf-8"), file_name="apex_footwear_upload_template.csv", mime="text/csv")

if data_mode == "Synthetic Apex demo":
    raw = generate_synthetic_data()
else:
    file = st.file_uploader("Upload Apex footwear sales file", type=["csv"])
    if file is None:
        st.info("Upload a CSV using the template above, or switch to synthetic demo.")
        st.stop()
    raw = pd.read_csv(file)
    missing = set(UPLOAD_COLUMNS).difference(raw.columns)
    if missing:
        st.error(f"Missing columns: {sorted(missing)}")
        st.stop()

df = preprocess(raw)
if store_filter != "All":
    df = df[df["store"] == store_filter]
if gender_filter != "All":
    df = df[df["gender"] == gender_filter]
if category_filter != "All":
    df = df[df["category"] == category_filter]

rec = forecast_and_recommend(df)
wall_sqft = WALL_WIDTH_FT * MERCH_HEIGHT_FT
kpis = compute_kpis(df, rec, wall_sqft)
plan = make_planogram(rec)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Monthly Sales", f"৳{kpis['monthly_sales']:,.0f}")
c2.metric("Sales / Sq Ft", f"৳{kpis['sales_per_sqft']:,.0f}", delta=f"{((kpis['sales_per_sqft'] / benchmark_sales_sqft) - 1) * 100:.1f}% vs benchmark")
c3.metric("GMROI", f"{kpis['gmroi']:.2f}")
c4.metric("OOS SKUs", kpis["oos_skus"])
c5.metric("OOD SKUs", kpis["ood_skus"])

st.markdown("### Planning logic")
st.write(textwrap.dedent("""
This app treats footwear wall space as a revenue asset. The 15 ft x 8 ft wall has only 6 ft usable merchandising height. The top 2 ft is reserved for Apex branding, category signage and campaign communication.

The planner gives prime wall space to products with stronger forecast demand, better GMROI, healthy stock and higher display urgency. It pushes weak, ageing and overstocked SKUs down the wall or into markdown action. It also highlights out-of-stock and out-of-display gaps so store teams can refill the wall before sales are lost.
"""))

st.markdown("### Recommended SKU actions")
st.dataframe(rec[["sku", "store", "product_line", "gender", "category", "colour", "size", "forecast_4w_units", "latest_stock_units", "latest_display_stock_units", "latest_gmroi", "latest_weeks_cover", "recommended_facings", "recommended_shelf_level", "recommended_action", "xai_reason"]], use_container_width=True)

st.markdown("### Wall planogram")
fig = draw_planogram(plan)
st.pyplot(fig, use_container_width=True)

st.markdown("### Store execution notes")
prime = rec[rec["recommended_action"] == "Prime wall placement"].head(5)["sku"].tolist() if not rec.empty else []
display_gap = int((rec["display_gap_units"] > 0).sum()) if not rec.empty else 0
st.write(textwrap.dedent(f"""
1. Execute bay-wise from left to right.
2. Keep prime SKUs refilled first: {', '.join(prime) if prime else 'No prime SKU found in current filter'}.
3. Display gap SKUs: {display_gap}. These must be refilled before store opening.
4. Do not merchandise above 6 ft. Use top 2 ft only for Apex branding and campaign signage.
5. Move weak and high-cover SKUs down the wall or put them into markdown review.
"""))

if use_ollama:
    st.markdown("### Ollama local explanation")
    if ollama_available(ollama_url):
        with st.spinner("Generating local explanation with Ollama..."):
            st.write(ask_ollama(ollama_url, ollama_model, kpis, rec))
    else:
        st.warning("Ollama is not reachable. Start Ollama locally and check model name.")

st.markdown("### Export")
st.download_button("Download recommendation CSV", data=rec.to_csv(index=False).encode("utf-8"), file_name="apex_footwear_planogram_recommendations.csv", mime="text/csv")
img = BytesIO()
fig.savefig(img, format="png", bbox_inches="tight")
img.seek(0)
st.download_button("Download planogram PNG", data=img, file_name="apex_footwear_wall_planogram.png", mime="image/png")

with st.expander("How to run locally"):
    st.code(textwrap.dedent("""
    pip install streamlit pandas numpy matplotlib requests
    streamlit run apex_footwear_planogram_agent.py

    Optional local AI explanation:
    ollama pull llama3.1:8b
    ollama serve
    """))

with st.expander("Production upgrades"):
    st.code(textwrap.dedent("""
    1. Connect POS, inventory and store display audit data.
    2. Add real forecast model by category, gender, size and campaign period.
    3. Add image-based wall compliance audit from store photos.
    4. Add store cluster rules for flagship, mall and high-street stores.
    5. Add automatic planogram emailing to store teams.
    6. Add markdown and transfer-to-outlet workflow for slow movers.
    """))
