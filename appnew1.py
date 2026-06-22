import math
import json
import textwrap
from io import BytesIO
from datetime import date, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
import requests

st.set_page_config(page_title="ApexSpace Pro", page_icon="👟", layout="wide")

# ============================================================
# ApexSpace Pro. Footwear Space Allocation + Wall Planogram Agent
# Replicates JuiceSpace Pro flow for footwear.
# Tabs: Dashboard, Explainable AI, Allocation, Planogram, Data, Schedule
# Wall: 15 ft x 8 ft. Merchandising only till 6 ft.
# ============================================================

WALL_WIDTH_FT = 15
WALL_HEIGHT_FT = 8
MERCH_HEIGHT_FT = 6
BAY_COUNT = 5
BAY_WIDTH_FT = WALL_WIDTH_FT / BAY_COUNT
LEVELS = ["Top Merch", "Eye Level", "Upper Mid", "Mid", "Lower", "Bottom"]

CATEGORIES = ["Formal Shoes", "Casual Shoes", "Sneakers", "Sandals", "Slippers", "School Shoes", "Bags"]
GENDERS = ["Men", "Women", "Kids"]
LINES = ["Apex", "Venturini", "Maverick", "Sprint", "Moochie", "Nino Rossi", "Twinkler", "Ipanema"]
COLORS = ["Black", "Brown", "Tan", "Navy", "White", "Beige", "Red", "Pink", "Blue", "Grey"]
SIZES = ["35", "36", "37", "38", "39", "40", "41", "42", "43", "44"]
STORES = ["Dhaka Flagship", "Gulshan Store", "Chattogram Store", "Outlet Store"]

CATEGORY_COLORS = {
    "Formal Shoes": "#2f2f2f",
    "Casual Shoes": "#8d6e63",
    "Sneakers": "#607d8b",
    "Sandals": "#e66b00",
    "Slippers": "#7c6bb0",
    "School Shoes": "#1b9e77",
    "Bags": "#d6278b",
}

BENCHMARKS = {
    "sales_per_sqft_monthly": 4500.0,
    "gmroi_target": 2.5,
    "turns_target_annual": 4.0,
    "weeks_cover_target": 8.0,
    "weeks_cover_high": 18.0,
    "display_fill_target": 0.90,
    "oos_target": 0,
    "ood_target": 0,
}

UPLOAD_COLUMNS = [
    "date", "store", "product_line", "gender", "category", "style", "colour", "size", "sku",
    "opening_stock_units", "receipts_units", "sales_units", "closing_stock_units",
    "display_capacity_units", "display_stock_units", "net_sales_value", "full_price",
    "avg_realized_price", "cost_per_unit", "gross_margin_pct", "markdown_pct",
    "age_days", "season_type", "last_year_sales_units"
]

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1500px;}
    section[data-testid="stSidebar"] {background:#f3f5f8;}
    .hero-title {font-size:42px; font-weight:800; color:#e65a00; margin-bottom:0px;}
    .hero-sub {color:#8a8d95; font-size:14px; margin-bottom:20px;}
    .pill {display:inline-block; padding:6px 12px; border-radius:18px; background:#e7f3ff; color:#1d6fd1; font-weight:700; font-size:12px;}
    .section-title {font-size:30px; font-weight:800; margin-top:18px; margin-bottom:14px;}
    .note {background:#fff7ed; border-left:5px solid #e65a00; padding:14px; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Data generation and template
# -----------------------------
def template_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": "2026-03-01",
            "store": "Dhaka Flagship",
            "product_line": "Apex",
            "gender": "Men",
            "category": "Sneakers",
            "style": "Men Casual Sneaker",
            "colour": "Black",
            "size": "42",
            "sku": "APX-SNK-BLK-42",
            "opening_stock_units": 18,
            "receipts_units": 4,
            "sales_units": 6,
            "closing_stock_units": 16,
            "display_capacity_units": 3,
            "display_stock_units": 2,
            "net_sales_value": 17940,
            "full_price": 3490,
            "avg_realized_price": 2990,
            "cost_per_unit": 1450,
            "gross_margin_pct": 0.52,
            "markdown_pct": 0.14,
            "age_days": 40,
            "season_type": "Core",
            "last_year_sales_units": 5,
        }
    ], columns=UPLOAD_COLUMNS)


@st.cache_data(show_spinner=False)
def generate_data(seed: int = 11, max_skus: int = 50) -> pd.DataFrame:
    """Generate lightweight synthetic data for Streamlit Cloud.
    50 SKU-store combinations x 52 weeks = 2,600 rows.
    This avoids the earlier 30 lakh+ row memory crash.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=52, freq="W-SUN")
    base_price = {
        "Formal Shoes": 3990, "Casual Shoes": 2990, "Sneakers": 3490,
        "Sandals": 1990, "Slippers": 1290, "School Shoes": 2490, "Bags": 2990,
    }
    cat_velocity = {
        "Formal Shoes": 0.85, "Casual Shoes": 1.05, "Sneakers": 1.18,
        "Sandals": 1.28, "Slippers": 1.10, "School Shoes": 0.80, "Bags": 0.55,
    }
    gender_factor = {"Men": 1.05, "Women": 1.12, "Kids": 0.88}
    store_factor = {"Dhaka Flagship": 1.20, "Gulshan Store": 1.05, "Chattogram Store": 0.92, "Outlet Store": 0.78}
    size_factor = {"35": .65, "36": .85, "37": 1.05, "38": 1.15, "39": 1.12, "40": 1.18, "41": 1.10, "42": 1.05, "43": .88, "44": .70, "NA": 1.0}

    # Build product-store universe, then sample it. Do not generate full Cartesian product.
    universe = []
    for store in STORES:
        for line in LINES:
            for gender in GENDERS:
                for category in CATEGORIES:
                    usable_sizes = ["NA"] if category == "Bags" else SIZES
                    # Keep only a practical color-size spread per category.
                    sampled_colours = rng.choice(COLORS, size=min(4, len(COLORS)), replace=False)
                    sampled_sizes = rng.choice(usable_sizes, size=min(4, len(usable_sizes)), replace=False)
                    for colour in sampled_colours:
                        for size in sampled_sizes:
                            universe.append((store, line, gender, category, colour, size))

    chosen_idx = rng.choice(len(universe), size=min(max_skus, len(universe)), replace=False)
    chosen = [universe[i] for i in chosen_idx]

    rows = []
    colour_map = {"Black":1.30,"Brown":1.08,"Tan":1.00,"Navy":.95,"White":.98,"Beige":.88,"Red":.75,"Pink":.78,"Blue":.93,"Grey":.90}

    for store, line, gender, category, colour, size in chosen:
        line_factor = rng.uniform(.85, 1.20)
        colour_factor = colour_map[colour]
        sku = f"{line[:3].upper()}-{category[:3].upper()}-{gender[:1]}-{colour[:3].upper()}-{size}"
        style = f"{gender} {category}"
        price = base_price[category] * rng.uniform(.92, 1.14)
        cost = price * rng.uniform(.35, .48)
        stock = int(rng.integers(8, 45))
        age = int(rng.integers(14, 210))
        season_type = rng.choice(["Core", "Fashion", "Carryover", "Clearance"], p=[.45,.25,.22,.08])
        ly_units_base = max(0, int(3.5 * cat_velocity[category] * gender_factor[gender] * line_factor * colour_factor * store_factor[store]))

        for dt in dates:
            month_factor = 1.12 if dt.month in [3,4,5,8,9,10] else .92
            md = float(rng.choice([0, .10, .15, .20, .30, .40], p=[.40,.18,.16,.12,.09,.05]))
            demand_mu = 2.8 * cat_velocity[category] * gender_factor[gender] * line_factor * colour_factor * store_factor[store] * month_factor
            demand_mu *= size_factor.get(size, 1.0)
            if season_type == "Clearance":
                demand_mu *= .65
            if md >= .20:
                demand_mu *= 1 + md * 1.3
            age_drag = max(.60, 1 - age / 520)
            sales = min(stock, int(rng.poisson(max(.1, demand_mu * age_drag))))
            receipts = int(rng.integers(0, 12)) if stock < 8 else int(rng.integers(0, 4))
            opening = stock
            closing = max(0, opening + receipts - sales)
            realized = price * (1 - md)
            display_cap = 2 if category != "Bags" else 1
            display_stock = int(min(closing, display_cap, max(0, rng.integers(0, display_cap + 1))))
            margin_pct = (realized - cost) / max(realized, 1)
            rows.append({
                "date": dt.date(), "store": store, "product_line": line, "gender": gender,
                "category": category, "style": style, "colour": colour, "size": size, "sku": sku,
                "opening_stock_units": opening, "receipts_units": receipts, "sales_units": sales,
                "closing_stock_units": closing, "display_capacity_units": display_cap, "display_stock_units": display_stock,
                "net_sales_value": round(sales * realized, 2), "full_price": round(price, 2),
                "avg_realized_price": round(realized, 2), "cost_per_unit": round(cost, 2),
                "gross_margin_pct": round(margin_pct, 3), "markdown_pct": md, "age_days": age,
                "season_type": season_type, "last_year_sales_units": ly_units_base,
            })
            stock = closing
            age += 7

    df = pd.DataFrame(rows)
    # Reduce memory footprint on Streamlit Cloud.
    for col in ["store", "product_line", "gender", "category", "style", "colour", "size", "sku", "season_type"]:
        df[col] = df[col].astype("category")
    return df

# -----------------------------
# Feature engineering and agent
# -----------------------------
@st.cache_data(show_spinner=False)
def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    for c in UPLOAD_COLUMNS:
        if c not in out.columns:
            out[c] = 0 if c not in ["date","store","product_line","gender","category","style","colour","size","sku","season_type"] else ""
    num_cols = [c for c in UPLOAD_COLUMNS if c not in ["date","store","product_line","gender","category","style","colour","size","sku","season_type"]]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    out["gross_margin_value"] = out["net_sales_value"] - out["sales_units"] * out["cost_per_unit"]
    out["avg_inventory_units"] = (out["opening_stock_units"] + out["closing_stock_units"]) / 2
    out["inventory_cost"] = out["avg_inventory_units"] * out["cost_per_unit"]
    out["gmroi"] = np.where(out["inventory_cost"] > 0, out["gross_margin_value"] / out["inventory_cost"], 0)
    out["inventory_turn"] = np.where(out["avg_inventory_units"] > 0, out["sales_units"] / out["avg_inventory_units"], 0)
    out["weeks_cover"] = np.where(out["sales_units"] > 0, out["closing_stock_units"] / out["sales_units"], 999)
    out["sell_through"] = np.where(out["opening_stock_units"] + out["receipts_units"] > 0, out["sales_units"] / (out["opening_stock_units"] + out["receipts_units"]), 0)
    out["oos_flag"] = (out["closing_stock_units"] <= 0).astype(int)
    out["ood_flag"] = (out["display_stock_units"] <= 0).astype(int)
    return out


@st.cache_data(show_spinner=False)
def latest_sku_view(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["store", "product_line", "gender", "category", "style", "colour", "size", "sku", "season_type"]
    g = df.sort_values("date").groupby(group_cols, as_index=False, observed=True).tail(12)
    agg = g.groupby(group_cols, as_index=False, observed=True).agg(
        weekly_sales_units=("sales_units", "mean"),
        last_12w_units=("sales_units", "sum"),
        net_sales_12w=("net_sales_value", "sum"),
        gross_margin_12w=("gross_margin_value", "sum"),
        avg_inventory_units=("avg_inventory_units", "mean"),
        inventory_cost=("inventory_cost", "mean"),
        closing_stock_units=("closing_stock_units", "last"),
        display_capacity_units=("display_capacity_units", "last"),
        display_stock_units=("display_stock_units", "last"),
        full_price=("full_price", "last"),
        avg_realized_price=("avg_realized_price", "last"),
        cost_per_unit=("cost_per_unit", "last"),
        markdown_pct=("markdown_pct", "last"),
        age_days=("age_days", "last"),
        last_year_sales_units=("last_year_sales_units", "last"),
    )
    agg["gmroi"] = np.where(agg["inventory_cost"] > 0, agg["gross_margin_12w"] / agg["inventory_cost"], 0)
    agg["inventory_turn_12w"] = np.where(agg["avg_inventory_units"] > 0, agg["last_12w_units"] / agg["avg_inventory_units"], 0)
    agg["weeks_cover"] = np.where(agg["weekly_sales_units"] > 0, agg["closing_stock_units"] / agg["weekly_sales_units"], 999)
    agg["display_fill"] = np.where(agg["display_capacity_units"] > 0, agg["display_stock_units"] / agg["display_capacity_units"], 0)
    agg["forecast_4w_units"] = np.maximum(0.1, (agg["weekly_sales_units"] * .65 + agg["last_year_sales_units"] * .35) * 4)
    return agg


def score_and_recommend(sku: pd.DataFrame, wall_sqft: float) -> pd.DataFrame:
    out = sku.copy()
    out["sales_density_score"] = out["net_sales_12w"] / max(wall_sqft, 1)
    out["velocity_score"] = out["weekly_sales_units"] * 8
    out["margin_score"] = out["gmroi"] * 18
    out["freshness_penalty"] = np.where(out["age_days"] > 180, -18, np.where(out["age_days"] > 120, -8, 0))
    out["stock_penalty"] = np.where(out["weeks_cover"] > BENCHMARKS["weeks_cover_high"], -15, 0)
    out["display_gap_boost"] = np.where(out["display_fill"] < BENCHMARKS["display_fill_target"], 8, 0)
    out["priority_score"] = (out["sales_density_score"] + out["velocity_score"] + out["margin_score"] + out["display_gap_boost"] + out["freshness_penalty"] + out["stock_penalty"]).clip(lower=1)

    def action(r):
        if r.closing_stock_units <= 0:
            return "Replenish immediately"
        if r.display_stock_units <= 0 and r.forecast_4w_units >= 2:
            return "Fix out-of-display"
        if r.weeks_cover > 22 and r.weekly_sales_units < 1.5:
            return "Reduce space / outlet transfer"
        if r.age_days > 180 and r.markdown_pct < .25:
            return "Markdown and clear"
        if r.gmroi > BENCHMARKS["gmroi_target"] and r.weekly_sales_units > 3:
            return "Increase visibility"
        return "Maintain"

    out["recommended_action"] = out.apply(action, axis=1)
    out["recommended_display_units"] = np.ceil(np.maximum(1, out["forecast_4w_units"] / 4 * 1.15)).astype(int).clip(1, 4)
    out["replenishment_gap"] = np.maximum(0, out["recommended_display_units"] - out["display_stock_units"]).astype(int)

    def level(r):
        if r.recommended_action in ["Increase visibility", "Fix out-of-display"] and r.category in ["Sneakers", "Sandals", "Casual Shoes"]:
            return "Eye Level"
        if r.category in ["Formal Shoes", "School Shoes"]:
            return "Upper Mid"
        if r.category in ["Slippers"]:
            return "Lower"
        if r.category == "Bags":
            return "Top Merch"
        return "Mid"

    out["recommended_level"] = out.apply(level, axis=1)
    return out.sort_values("priority_score", ascending=False)


def explain_row(r) -> str:
    reasons = []
    if r.weekly_sales_units >= 3:
        reasons.append("good sales velocity")
    if r.gmroi >= BENCHMARKS["gmroi_target"]:
        reasons.append("GMROI above target")
    if r.display_stock_units <= 0:
        reasons.append("out-of-display risk")
    if r.closing_stock_units <= 0:
        reasons.append("out-of-stock risk")
    if r.weeks_cover > BENCHMARKS["weeks_cover_high"]:
        reasons.append("high weeks of cover")
    if r.age_days > 180:
        reasons.append("aged stock")
    if not reasons:
        reasons.append("balanced space recommendation")
    return ", ".join(reasons)


def make_planogram(reco: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wall_width_in = WALL_WIDTH_FT * 12
    bay_width_in = wall_width_in / BAY_COUNT
    sku_reco = reco.copy().head(90)
    # Category bay strategy based on Apex style wall display
    category_bay = {
        "Formal Shoes": 1,
        "Casual Shoes": 2,
        "Sneakers": 3,
        "Sandals": 4,
        "Slippers": 5,
        "School Shoes": 1,
        "Bags": 5,
    }
    level_y = {"Bottom": 6, "Lower": 18, "Mid": 30, "Upper Mid": 42, "Eye Level": 54, "Top Merch": 66}
    slots = []
    slot_w = 9  # 20 products visible per row across 15 ft
    for _, r in sku_reco.iterrows():
        bay = category_bay.get(r.category, 3)
        preferred_level = r.recommended_level
        count = int(max(1, min(3, r.recommended_display_units)))
        for _ in range(count):
            slots.append({
                "sku": r.sku, "category": r.category, "product_line": r.product_line, "gender": r.gender,
                "colour": r.colour, "size": r.size, "bay": bay, "level": preferred_level,
                "priority_score": r.priority_score, "weekly_sales_units": r.weekly_sales_units,
                "gmroi": r.gmroi, "action": r.recommended_action,
            })
    plan = pd.DataFrame(slots).sort_values(["bay", "level", "priority_score"], ascending=[True, True, False]).reset_index(drop=True)
    if plan.empty:
        return plan, pd.DataFrame()

    used = {}
    positions = []
    sku_codes = {}
    for i, sku in enumerate(plan["sku"].unique(), start=1):
        sku_codes[sku] = f"S{i:03d}"
    for _, r in plan.iterrows():
        key = (int(r.bay), r.level)
        used.setdefault(key, 0)
        x = (int(r.bay) - 1) * bay_width_in + used[key] * slot_w
        if x + slot_w > int(r.bay) * bay_width_in:
            # Try another level in same bay
            placed = False
            for lev in LEVELS:
                key2 = (int(r.bay), lev)
                used.setdefault(key2, 0)
                x2 = (int(r.bay) - 1) * bay_width_in + used[key2] * slot_w
                if x2 + slot_w <= int(r.bay) * bay_width_in:
                    key = key2; x = x2; placed = True; break
            if not placed:
                continue
        used[key] += 1
        positions.append({**r.to_dict(), "sku_code": sku_codes[r.sku], "x_in": x, "y_in": level_y[key[1]], "w_in": slot_w, "h_in": 9})
    planogram = pd.DataFrame(positions)
    legend = planogram[["sku_code", "sku", "product_line", "gender", "category", "colour", "size", "bay", "level", "action", "weekly_sales_units", "gmroi"]].drop_duplicates("sku_code")
    return planogram, legend


def draw_wall(planogram: pd.DataFrame, legend: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(24, 12))
    wall_w, wall_h, merch_h = WALL_WIDTH_FT * 12, WALL_HEIGHT_FT * 12, MERCH_HEIGHT_FT * 12
    bay_w = wall_w / BAY_COUNT

    # Outer wall and merchandising boundary
    ax.add_patch(patches.Rectangle((0, 0), wall_w, wall_h, fill=False, linewidth=2, edgecolor="#222"))
    ax.add_patch(patches.Rectangle((0, merch_h), wall_w, wall_h - merch_h, facecolor="#f5f5f5", edgecolor="#222", linewidth=1.5))
    ax.text(wall_w / 2, merch_h + 14, "APEX BRAND HEADER / CAMPAIGN AREA. No merchandising above 6 ft", ha="center", va="center", fontsize=16, fontweight="bold")

    for b in range(BAY_COUNT + 1):
        ax.plot([b * bay_w, b * bay_w], [0, wall_h], color="#9e9e9e", linewidth=1.4)
    for y in [6, 18, 30, 42, 54, 66, 72]:
        ax.plot([0, wall_w], [y, y], color="#222", linewidth=1.6 if y <= merch_h else .8)
    for b in range(1, BAY_COUNT + 1):
        ax.text((b - .5) * bay_w, 2, f"Bay {b}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    for level, y in {"Bottom":6,"Lower":18,"Mid":30,"Upper Mid":42,"Eye Level":54,"Top Merch":66}.items():
        ax.text(-7, y + 4, level, ha="right", va="center", fontsize=10, fontweight="bold")

    if not planogram.empty:
        for _, r in planogram.iterrows():
            face = CATEGORY_COLORS.get(r.category, "#cccccc")
            rect = patches.Rectangle((r.x_in + .4, r.y_in + .8), r.w_in - .8, r.h_in, facecolor=face, edgecolor="black", linewidth=1)
            ax.add_patch(rect)
            text_color = "white" if r.category in ["Formal Shoes", "Casual Shoes", "Sneakers", "Slippers", "Bags"] else "black"
            # Large readable label. SKU code inside, full mapping below in legend table.
            ax.text(r.x_in + r.w_in / 2, r.y_in + 6.0, f"{r.sku_code}\n{str(r.size)}", ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)

    # Category legend inside chart
    x0 = wall_w + 5
    y0 = wall_h - 5
    ax.text(x0, y0, "Category legend", fontsize=12, fontweight="bold", va="top")
    for i, (cat, color) in enumerate(CATEGORY_COLORS.items()):
        yy = y0 - 7 - i * 6
        ax.add_patch(patches.Rectangle((x0, yy - 3), 5, 3.5, facecolor=color, edgecolor="black"))
        ax.text(x0 + 7, yy - 1, cat, fontsize=10, va="center")

    ax.set_xlim(-14, wall_w + 65)
    ax.set_ylim(0, wall_h + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Apex Footwear Wall Planogram. 15 ft x 8 ft. Merchandising till 6 ft only", fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def kpis(df: pd.DataFrame, reco: pd.DataFrame) -> dict:
    latest = df.sort_values("date").groupby("sku", as_index=False, observed=True).tail(12)
    monthly_sales = latest["net_sales_value"].sum() / 3
    sales_sqft = monthly_sales / (WALL_WIDTH_FT * MERCH_HEIGHT_FT)
    gross_margin = latest["gross_margin_value"].sum()
    inv_cost = latest["inventory_cost"].mean()
    gmroi = gross_margin / max(inv_cost, 1)
    turns = latest["sales_units"].sum() / max(latest["avg_inventory_units"].mean(), 1)
    return {
        "monthly_sales": monthly_sales,
        "sales_sqft": sales_sqft,
        "gmroi": gmroi,
        "turns": turns,
        "oos": int((reco["closing_stock_units"] <= 0).sum()),
        "ood": int((reco["display_stock_units"] <= 0).sum()),
        "uplift": 4.5 + min(4.0, int((reco["display_stock_units"] <= 0).sum()) / 40),
    }


def ask_ollama(base_url: str, model: str, metrics: dict, sample: pd.DataFrame) -> str:
    rows = sample.head(15)[["sku", "category", "weekly_sales_units", "gmroi", "weeks_cover", "display_stock_units", "recommended_action", "recommended_level", "xai_reason"]].to_dict(orient="records")
    prompt = f"""
You are a retail footwear space planning expert. Explain this Apex footwear wall planogram decision in plain English.
Mention sales per sq ft, GMROI, OOS, OOD, allocation logic, and what store teams should execute.
Keep under 250 words.
Metrics: {json.dumps(metrics, indent=2)}
Top recommendations: {json.dumps(rows, indent=2)}
""".strip()
    try:
        r = requests.post(f"{base_url}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=45)
        r.raise_for_status()
        return r.json().get("response", "No response from Ollama.")
    except Exception as e:
        return f"Ollama not available or failed: {e}"

# -----------------------------
# Sidebar and data load
# -----------------------------
with st.sidebar:
    st.markdown("## 👟 ApexSpace Pro")
    st.caption("Explainable AI · Footwear Space Allocation Agent")
    st.divider()
    st.markdown("### STORE PROFILE")
    data_mode = st.radio("Data", ["Synthetic data", "Upload real data"], index=0)
    store_filter = st.selectbox("Store", ["All"] + STORES, index=0)
    category_filter = st.selectbox("Category", ["All"] + CATEGORIES, index=0)
    gender_filter = st.selectbox("Gender", ["All"] + GENDERS, index=0)
    wall_width_ft = 15
st.number_input("Wall width ft", value=wall_width_ft, disabled=True)
wall_height_ft = 8
st.number_input("Wall height ft", value=wall_height_ft, disabled=True)
merch_height_ft = 6
st.number_input("Merchandising height ft", value=merch_height_ft, disabled=True)
st.divider()
st.markdown("### LOCAL AI")
use_ollama = st.checkbox("Use Ollama explanation", value=False)
ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
ollama_model = st.text_input("Model", "llama3.1:8b")

st.markdown('<div class="hero-title">👟 ApexSpace Pro <span style="font-size:16px;color:#999;">v2 · Explainable AI Edition</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Apex footwear · Multi-store wall planner · 15 ft x 8 ft wall · merchandising till 6 ft · {date.today().strftime("%d %b %Y")}</div>', unsafe_allow_html=True)

if data_mode == "Synthetic data":
    raw = generate_data()
    st.markdown('<span class="pill">SYNTHETIC DATA</span> &nbsp; Switch to the Data tab to upload real data.', unsafe_allow_html=True)
else:
    uploaded = st.file_uploader("Upload Apex footwear CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV using the template in the Data tab, or switch to synthetic data.")
        st.stop()
    raw = pd.read_csv(uploaded)
    missing = sorted(set(UPLOAD_COLUMNS) - set(raw.columns))
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

raw = prepare(raw)
if store_filter != "All": raw = raw[raw["store"] == store_filter]
if category_filter != "All": raw = raw[raw["category"] == category_filter]
if gender_filter != "All": raw = raw[raw["gender"] == gender_filter]

sku = latest_sku_view(raw)
reco = score_and_recommend(sku, WALL_WIDTH_FT * MERCH_HEIGHT_FT)
reco["xai_reason"] = reco.apply(explain_row, axis=1)
planogram, legend = make_planogram(reco)
metrics = kpis(raw, reco)

tab_dashboard, tab_xai, tab_alloc, tab_planogram, tab_data, tab_schedule = st.tabs(["📊 Dashboard", "🧠 cd plano AI", "🗂️ Allocation", "📐 Planogram", "📥 Data", "🗓️ Schedule"])

with tab_dashboard:
    st.markdown('<div class="section-title">📊 Performance Dashboard</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Monthly sales", f"₹{metrics['monthly_sales']/100000:.1f}L")
    c2.metric("Avg sales/sqft", f"₹{metrics['sales_sqft']:,.0f}", f"vs ₹{BENCHMARKS['sales_per_sqft_monthly']:,.0f} benchmark")
    c3.metric("Portfolio GMROI", f"{metrics['gmroi']:.2f}x", f"Target {BENCHMARKS['gmroi_target']}x")
    c4.metric("Projected uplift", f"+{metrics['uplift']:.1f}%", "new vs current")
    c5.metric("OOS / OOD SKUs", f"{metrics['oos']} / {metrics['ood']}", "target zero")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Monthly sales by category")
        chart = raw.groupby("category", as_index=False, observed=True)["net_sales_value"].sum().sort_values("net_sales_value", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(chart["category"], chart["net_sales_value"])
        ax.set_xlabel("Category")  
        ax.set_ylabel("Net Sales Value")
        ax.set_title("Category Sales")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig, use_container_width=True)
    with col_b:
        st.markdown("#### GMROI vs category")
        gm = raw.groupby("category", as_index=False, observed=True).agg(gross_margin=("gross_margin_value","sum"), inv=("inventory_cost","mean"))
        gm["gmroi"] = gm["gross_margin"] / gm["inv"].clip(lower=1)
        fig2, ax2 = plt.subplots(figsize=(10,4))
        ax2.bar(gm["category"], gm["gmroi"])
        ax2.set_xlabel("Category")
        ax2.set_ylabel("GMROI")
        ax2.set_title("GMROI by Category")
        ax2.tick_params(axis="x", rotation=45)
        st.pyplot(fig2, use_container_width=True)

    st.markdown("#### What changed")
    st.markdown("""
    <div class="note">
    The agent prioritises high velocity and high GMROI footwear on eye and upper-mid levels, fixes out-of-display gaps, reduces space for slow aged stock, and keeps Apex wall execution centrally controlled but customized by store performance.
    </div>
    """, unsafe_allow_html=True)

with tab_xai:
    st.markdown('<div class="section-title">🧠 Explainable AI</div>', unsafe_allow_html=True)
    st.write("Each recommendation is explained using sales velocity, GMROI, weeks cover, display stock, age and current inventory position.")
    xai_cols = ["sku", "product_line", "gender", "category", "colour", "size", "weekly_sales_units", "gmroi", "weeks_cover", "display_stock_units", "closing_stock_units", "recommended_action", "recommended_level", "xai_reason"]
    st.dataframe(reco[xai_cols].head(300), use_container_width=True, height=520)

    if use_ollama:
        st.markdown("#### Local Ollama explanation")
        with st.spinner("Calling Ollama locally..."):
            st.write(ask_ollama(ollama_url, ollama_model, metrics, reco))
    else:
        st.info("Turn on Ollama in the sidebar for local natural-language explanation.")

with tab_alloc:
    st.markdown('<div class="section-title">🗂️ Allocation</div>', unsafe_allow_html=True)
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Increase visibility", int((reco["recommended_action"] == "Increase visibility").sum()))
    a2.metric("Fix OOD", int((reco["recommended_action"] == "Fix out-of-display").sum()))
    a3.metric("Replenish", int((reco["recommended_action"] == "Replenish immediately").sum()))
    a4.metric("Reduce / transfer", int((reco["recommended_action"] == "Reduce space / outlet transfer").sum()))
    st.dataframe(reco[["sku", "category", "gender", "colour", "size", "forecast_4w_units", "closing_stock_units", "display_stock_units", "recommended_display_units", "replenishment_gap", "recommended_level", "recommended_action"]].head(500), use_container_width=True, height=560)

with tab_planogram:
    st.markdown('<div class="section-title">📐 Wall Planogram</div>', unsafe_allow_html=True)
    st.caption("Readable SKU codes are printed inside the planogram. Full SKU details are in the legend table below.")
    fig = draw_wall(planogram, legend)
    st.pyplot(fig, use_container_width=True)

    img = BytesIO()
    fig.savefig(img, format="png", dpi=180, bbox_inches="tight")
    img.seek(0)
    st.download_button("Download planogram PNG", img, "apex_footwear_wall_planogram.png", "image/png")

    st.markdown("#### SKU legend for store execution")
    st.dataframe(legend.sort_values(["bay", "level", "sku_code"]), use_container_width=True, height=420)
    st.download_button("Download SKU legend CSV", legend.to_csv(index=False).encode("utf-8"), "apex_planogram_sku_legend.csv", "text/csv")

with tab_data:
    st.markdown('<div class="section-title">📥 Data</div>', unsafe_allow_html=True)
    st.markdown("#### Upload template")
    t = template_df()
    st.dataframe(t, use_container_width=True)
    st.download_button("Download upload template", t.to_csv(index=False).encode("utf-8"), "apex_footwear_upload_template.csv", "text/csv")
    st.markdown("#### Current data preview")
    st.dataframe(raw.head(500), use_container_width=True, height=420)
    st.download_button("Download synthetic/current data", raw.to_csv(index=False).encode("utf-8"), "apex_footwear_sales_data.csv", "text/csv")

with tab_schedule:
    st.markdown('<div class="section-title">🗓️ Schedule</div>', unsafe_allow_html=True)
    st.write("Use the schedule tab to decide how frequently the wall should refresh by product type.")
    schedule = pd.DataFrame([
        {"Product type": "New launch / campaign footwear", "Refresh": "Daily", "Reason": "Early sales signal decides visibility fast"},
        {"Product type": "Core sneakers and sandals", "Refresh": "Weekly", "Reason": "Enough movement to update facings and levels"},
        {"Product type": "Formal shoes", "Refresh": "Weekly / Fortnightly", "Reason": "Slower demand cycle"},
        {"Product type": "School shoes", "Refresh": "Daily during back-to-school", "Reason": "Seasonal stock-out risk"},
        {"Product type": "Aged / clearance stock", "Refresh": "Weekly", "Reason": "Transfer or markdown decision"},
        {"Product type": "Bags", "Refresh": "Monthly", "Reason": "Lower velocity, less wall reset frequency"},
    ])
    st.dataframe(schedule, use_container_width=True)
    st.markdown("#### Suggested automation")
    st.code(textwrap.dedent("""
    # Run manually
    uv run streamlit run app.py

    # Production scheduler idea
    # Daily: ingest POS + inventory + display stock
    # Weekly: regenerate wall allocation and planogram
    # Monthly: review category benchmarks and range rationalisation
    """))

st.divider()
st.caption("ApexSpace Pro. Synthetic model for retail planning demonstration. Replace with POS, ERP, WMS and store display data for production.")
