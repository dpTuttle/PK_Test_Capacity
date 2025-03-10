import os
import io
import math
import base64
import csv

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly for bullet/gauge charts (App2)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure 'uploads' and 'static' directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)

# Global data
capacity_df = pd.DataFrame()
demand_df = pd.DataFrame()

# -------------- Shared Helper Functions --------------

def compute_capacity(row):
    """
    For each row in a capacity_data.xlsx:
      monthly capacity = min(labor / cycle_time, equipment * equip_throughput) * hours_per_month
    """
    hours_per_month = 160  # or user-defined
    equip_throughput = 1.0
    labor_capacity = (row["Labor Headcount"] / row["Cycle Time (hr/unit)"]) * hours_per_month
    equipment_capacity = (row["Equipment"] * equip_throughput) * hours_per_month
    return min(labor_capacity, equipment_capacity)

class Process:
    def __init__(self, name, cycle_time, labor, equipment):
        self.name = name
        self.cycle_time = float(cycle_time)
        self.labor = float(labor)
        self.equipment = float(equipment)

    def capacity_per_period(self, hours_per_period, equip_throughput):
        labor_capacity = (self.labor / self.cycle_time) * hours_per_period
        equipment_capacity = (self.equipment * equip_throughput) * hours_per_period
        return min(labor_capacity, equipment_capacity)

def get_days_for_timescale(timescale):
    """Return how many days define 'one period' based on timescale."""
    if timescale == "annual":
        return 365
    elif timescale == "monthly":
        return 30
    elif timescale == "weekly":
        return 7
    elif timescale == "daily":
        return 1
    return 30  # default monthly

def generate_heatmap(df, value_col):
    """Generate a heatmap from DataFrame columns: [Process, value_col]."""
    if df.empty:
        return ""
    pivot_df = df.set_index("Process")[[value_col]]
    plt.figure(figsize=(5, 3))
    sns.heatmap(pivot_df, annot=True, cmap="Reds", fmt=".1f")
    plt.title("Process Capacities Heatmap")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def build_csv(results):
    """Convert a list of dicts to a CSV string."""
    if not results:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()

# ------------------- App1 Routes -------------------

@app.route("/upload_excel", methods=["POST"])
def upload_excel():
    """
    Example route to handle an Excel file for "App1" usage.
    Expects columns: Process, Cycle Time (hr/unit), Labor Headcount, Equipment
    Returns JSON with parsed data.
    """
    file = request.files.get('excelFile')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    try:
        df = pd.read_excel(filepath, sheet_name=0)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Convert each row to dict
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "name": row["Process"],
            "cycle_time": row["Cycle Time (hr/unit)"],
            "labor": row["Labor Headcount"],
            "equipment": row["Equipment"]
        })
    return jsonify({"processData": data_list})


@app.route("/calculate", methods=["POST"])
def calculate():
    """
    Example route for "App1" scenario logic:
      1) SHIFT SCHEDULING
      2) TIME SCALE
      3) eqp throughput
      4) build Process objects
      5) scenario: max_throughput or desired_units
    """
    req = request.json
    if not req:
        return jsonify({"error": "No data"}), 400

    # SHIFT + TIME
    shifts_per_day = float(req.get("shifts_per_day", 1))
    hours_per_shift = float(req.get("hours_per_shift", 8))
    timescale = req.get("timescale", "monthly")
    days_per_period = get_days_for_timescale(timescale)
    hours_per_period = shifts_per_day * hours_per_shift * days_per_period
    equip_throughput = float(req.get("equip_throughput", 1.0))

    # Build processes
    process_list = req.get("processes", [])
    processes = []
    for p in process_list:
        obj = Process(p["name"], p["cycle_time"], p["labor"], p["equipment"])
        processes.append(obj)

    scenario = req.get("scenario")

    if scenario == "max_throughput":
        # Calculate capacities
        capacities = []
        for proc in processes:
            cap = proc.capacity_per_period(hours_per_period, equip_throughput)
            capacities.append({"Process": proc.name, "Capacity": round(cap, 1)})
        if not capacities:
            return jsonify({"message": "No processes defined"}), 200

        min_cap = min([c["Capacity"] for c in capacities])
        bottleneck = [c for c in capacities if c["Capacity"] == min_cap][0]["Process"]
        df_cap = pd.DataFrame(capacities)
        heatmap_img = generate_heatmap(df_cap, "Capacity")
        msg = f"Based on current conditions, your max {timescale} throughput is {min_cap:.1f} units. Bottleneck: {bottleneck}."
        return jsonify({
            "scenario": scenario,
            "capacities": capacities,
            "message": msg,
            "heatmap": heatmap_img
        })

    elif scenario == "desired_units":
        # We want to produce X units
        desired_units = float(req.get("desired_units", 1000))
        results = []
        for proc in processes:
            current_cap = proc.capacity_per_period(hours_per_period, equip_throughput)
            if current_cap >= desired_units:
                note = "No changes"
                needed_labor = proc.labor
                needed_equipment = proc.equipment
            else:
                needed_labor_calc = (desired_units * proc.cycle_time) / hours_per_period
                needed_equipment_calc = math.ceil(desired_units / (equip_throughput * hours_per_period))
                needed_labor = max(proc.labor, needed_labor_calc)
                needed_equipment = max(proc.equipment, needed_equipment_calc)
                note = "Adjusted"
            results.append({
                "Process": proc.name,
                "Current Labor": proc.labor,
                "Current Equipment": proc.equipment,
                "Needed Labor": round(needed_labor, 2),
                "Needed Equipment": round(needed_equipment, 2),
                "Note": note
            })
        csv_str = build_csv(results)
        return jsonify({
            "scenario": scenario,
            "results": results,
            "csv_data": csv_str
        })

    return jsonify({"error": "Invalid scenario"}), 400


@app.route("/download_csv", methods=["POST"])
def download_csv():
    """POST with 'csv_data' to receive a file download."""
    csv_data = request.form.get("csv_data", "")
    buf = io.BytesIO(csv_data.encode('utf-8'))
    return send_file(buf, as_attachment=True, download_name="ideal_setup.csv", mimetype="text/csv")


# ------------------- App2 Routes -------------------

@app.route("/upload_capacity", methods=["POST"])
def upload_capacity():
    """
    Another route for uploading capacity (for the bullet/gauge logic).
    If you want to unify these routes with /upload_excel, do so. 
    If you need them separate, keep them separate & rename them carefully.
    """
    global capacity_df

    file = request.files.get("capacityFile")
    if not file:
        return jsonify({"error": "No capacity file provided"}), 400
    
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    df["Capacity"] = df.apply(compute_capacity, axis=1)
    capacity_df = df
    return jsonify({"message": "Capacity data uploaded and computed.", "data": df.to_dict(orient="records")})

@app.route("/upload_demand", methods=["POST"])
def upload_demand():
    """
    Another route for demand data.
    Expects columns: Process, Demand (Units/Month)
    """
    global demand_df

    file = request.files.get("demandFile")
    if not file:
        return jsonify({"error": "No demand file provided"}), 400
    
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    demand_df = df
    return jsonify({"message": "Demand data uploaded.", "data": df.to_dict(orient="records")})

@app.route("/visuals")
def visuals():
    """
    Renders bullet/gauge charts for capacity vs. demand.
    """
    if capacity_df.empty or demand_df.empty:
        return "<h3>Please upload capacity and demand data first!</h3>"

    merged = pd.merge(capacity_df, demand_df, on="Process", how="inner")
    html_parts = ["<h2>Capacity vs Demand Dashboard</h2>"]
    for _, row in merged.iterrows():
        process_name = row["Process"]
        capacity_val = row["Capacity"]
        demand_val = row["Demand (Units/Month)"]

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.5, 0.5],
            horizontal_spacing=0.15,
            specs=[[{"type": "indicator"}, {"type": "indicator"}]],
            subplot_titles=[f"{process_name} - Bullet", f"{process_name} - Gauge"]
        )

        # Bullet Chart
        max_axis = max(capacity_val, demand_val) * 1.2
        bullet = go.Indicator(
            mode="number+gauge",
            value=demand_val,
            gauge={
                "shape": "bullet",
                "axis": {"range": [None, max_axis]},
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.8,
                    "value": capacity_val
                },
                "bar": {"color": "#1f77b4"},
            },
            title={"text": "Demand vs Capacity"},
            domain={'row': 0, 'column': 0}
        )

        # Gauge Chart
        utilization_pct = (demand_val / capacity_val * 100) if capacity_val > 0 else 0
        gauge = go.Indicator(
            mode="gauge+number+delta",
            value=utilization_pct,
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 200]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 80], 'color': '#c7e9c0'},
                    {'range': [80, 100], 'color': '#ffffbf'},
                    {'range': [100, 150], 'color': '#fdaf61'},
                    {'range': [150, 200], 'color': '#f03b20'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            },
            title={"text": "Utilization (%)"},
            domain={'row': 0, 'column': 1}
        )

        fig.add_trace(bullet, row=1, col=1)
        fig.add_trace(gauge, row=1, col=2)
        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=40, r=40, t=70, b=40),
            title=dict(text=f"{process_name}", x=0.5, xanchor='center'),
            font=dict(family="Arial", size=12),
        )
        chart_html = fig.to_html(full_html=False)
        html_parts.append(f"<div style='margin-bottom:30px; border:1px solid #ddd; padding:10px;'>{chart_html}</div>")

    return "".join(html_parts)

# ------------------- Combined Index (Tabs, etc.) -------------------

@app.route("/")
def combined_index():
    """
    If you want a single tab-based page for both sets of functionality,
    you can serve a 'combined_index.html' that has multiple tabs or iframes.
    For now, let's just serve a single 'index.html'.
    """
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5000)
