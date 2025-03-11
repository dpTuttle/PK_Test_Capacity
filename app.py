import os
import io
import math
import base64
import csv
from datetime import datetime
from calendar import monthrange

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd

# Matplotlib and Seaborn for Heatmaps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Plotly for Bullet/Gauge Charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure 'uploads' and 'static' directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)

# Global Data
capacity_df = pd.DataFrame()
demand_df = pd.DataFrame()
forecast_demand_df = pd.DataFrame()  # For Tab 3
uploaded_process_df = pd.DataFrame()  # For storing uploaded Excel data from Tab 1

# Stores final capacity results from Tab 1 (App1) scenario
app1_capacity_result = {}

# ------------------- Helper Functions -------------------

def get_days_for_timescale(timescale):
    if timescale == "annual":
        return 365
    elif timescale == "monthly":
        now = datetime.now()
        return monthrange(now.year, now.month)[1]
    elif timescale == "weekly":
        return 7
    elif timescale == "daily":
        return 1
    return 30  # Default

def build_csv(results):
    if not results:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()

def generate_heatmap(df, value_col):
    if df.empty:
        return ""
    pivot_df = df.set_index("Process")[[value_col]]
    plt.figure(figsize=(5, 3))
    sns.heatmap(pivot_df, annot=True, cmap="Reds", fmt=".1f")
    plt.title("Process Capacities Heatmap")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def compute_capacity(row):
    # Use a fixed 720 hours per month (30 days * 24 hours) for example purposes.
    hours_per_month = 720  
    labor_capacity = (row["Labor Headcount"] / row["Cycle Time (hr/unit)"]) * hours_per_month
    return labor_capacity

# ------------------- Process Class -------------------

class Process:
    """
    Each process has: name, cycle_time, labor, bsc, incubator.
    Capacity is additionally constrained by the fact that each process
    can only be run once per day per room.
    """
    def __init__(self, name, cycle_time, labor, bsc, incubator):
        self.name = name
        self.cycle_time = float(cycle_time)
        self.labor = float(labor)
        self.bsc = float(bsc)
        self.incubator = float(incubator)

    def capacity_per_period(self, hours_per_period, max_bsc, max_incubators, days_per_period, rooms):
        labor_cap = (self.labor / self.cycle_time) * hours_per_period

        if self.bsc > 0:
            bsc_cap = (max_bsc / self.bsc) * (hours_per_period / self.cycle_time)
        else:
            bsc_cap = 1e9

        if self.incubator > 0:
            inc_cap = (max_incubators / self.incubator) * (hours_per_period / self.cycle_time)
        else:
            inc_cap = 1e9

        # Each room can only process the process once per day.
        room_cap = days_per_period * rooms

        return min(labor_cap, bsc_cap, inc_cap, room_cap)

# ------------------- Tab 1 (App1) Routes -------------------

@app.route("/upload_excel", methods=["POST"])
def upload_excel():
    """
    Expects an Excel file with columns:
      - Process
      - Cycle Time (hr/unit)
      - Labor Headcount
      - BSC
      - Incubator
    Returns JSON with a key 'processData' that maps to a list of records
    with keys: name, cycle_time, labor, bsc, incubator.
    """
    global uploaded_process_df

    file = request.files.get('excelFile')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Map Excel columns to expected keys for the front end.
    process_data = []
    for _, row in df.iterrows():
        process_data.append({
            "name": row["Process"],
            "cycle_time": row["Cycle Time (hr/unit)"],
            "labor": row["Labor Headcount"],
            "bsc": row["BSC"],
            "incubator": row["Incubator"]
        })

    uploaded_process_df = df  # Store globally if needed

    return jsonify({
        "message": "Excel data uploaded successfully!",
        "processData": process_data
    })

@app.route("/calculate", methods=["POST"])
def calculate():
    global app1_capacity_result

    req = request.json
    if not req:
        return jsonify({"error": "No data"}), 400

    shifts_per_day = float(req.get("shifts_per_day", 1))
    hours_per_shift = float(req.get("hours_per_shift", 8))
    timescale = req.get("timescale", "monthly")
    days_per_period = get_days_for_timescale(timescale)
    hours_per_period = shifts_per_day * hours_per_shift * days_per_period

    print(f"Timescale: {timescale}, Days per period: {days_per_period}, Hours per period: {hours_per_period}")

    rooms = float(req.get("rooms", 1))
    max_bsc = float(req.get("max_bsc", 1))
    max_incubators = float(req.get("max_incubators", 1))
    scenario = req.get("scenario")
    desired_units = float(req.get("desired_units", 1000))

    process_list = req.get("processes", [])
    processes = []
    for p in process_list:
        try:
            obj = Process(p["name"], p["cycle_time"], p["labor"], p["bsc"], p["incubator"])
            processes.append(obj)
        except Exception as e:
            return jsonify({"error": f"Error processing process data: {e}"}), 400

    if scenario == "max_throughput":
        capacities = []
        for proc in processes:
            cap = proc.capacity_per_period(
                hours_per_period, 
                max_bsc * rooms, 
                max_incubators * rooms,
                days_per_period,
                rooms
            )
            capacities.append({
                "Process": proc.name,
                "Capacity": round(cap, 1)
            })

        if not capacities:
            return jsonify({"message": "No processes defined"}), 200

        min_cap = min(c["Capacity"] for c in capacities)
        bottleneck = [c for c in capacities if c["Capacity"] == min_cap][0]["Process"]

        df_cap = pd.DataFrame(capacities)
        heatmap_img = generate_heatmap(df_cap, "Capacity")

        msg = (f"Max {timescale} throughput is {min_cap:.1f} units, with {rooms} room(s), "
               f"{max_bsc} BSC(s), {max_incubators} Incubator(s). Bottleneck: {bottleneck}.")

        app1_capacity_result = {
            "rooms": rooms,
            "max_bsc": max_bsc,
            "max_incubators": max_incubators,
            "timescale": timescale,
            "overall_capacity": min_cap,
            "details": capacities
        }

        return jsonify({
            "scenario": scenario,
            "capacities": capacities,
            "message": msg,
            "heatmap": heatmap_img
        })

    elif scenario == "desired_units":
        results = []
        for proc in processes:
            cap = proc.capacity_per_period(
                hours_per_period, 
                max_bsc * rooms, 
                max_incubators * rooms,
                days_per_period,
                rooms
            )
            if cap >= desired_units:
                note = "No changes"
                needed_labor = proc.labor
                needed_bsc = proc.bsc
                needed_inc = proc.incubator
            else:
                note = "Adjusted"
                factor = desired_units / cap if cap > 0 else 1
                needed_labor = max(proc.labor, proc.labor * factor)
                needed_bsc    = max(proc.bsc,    proc.bsc * factor)
                needed_inc    = max(proc.incubator, proc.incubator * factor)

            results.append({
                "Process": proc.name,
                "Rooms": rooms,
                "Max_BSC": max_bsc,
                "Max_Incubators": max_incubators,
                "Current Labor": proc.labor,
                "Current BSC": proc.bsc,
                "Current Incubator": proc.incubator,
                "Needed Labor": round(needed_labor, 2),
                "Needed BSC": round(needed_bsc, 2),
                "Needed Incubator": round(needed_inc, 2),
                "Note": note
            })

        csv_str = build_csv(results)
        app1_capacity_result = {
            "rooms": rooms,
            "max_bsc": max_bsc,
            "max_incubators": max_incubators,
            "timescale": timescale,
            "desired_units": desired_units,
            "details": results
        }

        return jsonify({
            "scenario": scenario,
            "results": results,
            "csv_data": csv_str
        })

    return jsonify({"error": "Invalid scenario"}), 400

@app.route("/download_csv", methods=["POST"])
def download_csv():
    csv_data = request.form.get("csv_data", "")
    buf = io.BytesIO(csv_data.encode('utf-8'))
    return send_file(buf, as_attachment=True, download_name="ideal_setup.csv", mimetype="text/csv")

# ------------------- Tab 2 (App2) Routes (Bullet/Gauge) -------------------

@app.route("/upload_capacity", methods=["POST"])
def upload_capacity():
    global capacity_df

    file = request.files.get("capacityFile")
    if not file:
        return jsonify({"error": "No capacity file provided"}), 400

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    df["Capacity"] = df.apply(compute_capacity, axis=1)
    capacity_df = df

    return jsonify({
        "message": "Capacity data uploaded and computed.",
        "data": df.to_dict(orient="records")
    })

@app.route("/upload_demand", methods=["POST"])
def upload_demand():
    global demand_df

    file = request.files.get("demandFile")
    if not file:
        return jsonify({"error": "No demand file provided"}), 400

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    demand_df = df

    return jsonify({
        "message": "Demand data uploaded.",
        "data": df.to_dict(orient="records")
    })

@app.route("/visuals")
def visuals():
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

        utilization_pct = (demand_val / capacity_val * 100) if capacity_val > 0 else 0
        gauge = go.Indicator(
            mode="gauge+number+delta",
            value=utilization_pct,
            delta={'reference': 100},
            gauge={
                'axis': {'range': [None, 200]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 80],   'color': '#c7e9c0'},
                    {'range': [80, 100], 'color': '#ffffbf'},
                    {'range': [100, 150],'color': '#fdaf61'},
                    {'range': [150, 200],'color': '#f03b20'}
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
        fig.add_trace(gauge,  row=1, col=2)

        fig.update_layout(
            template="plotly_white",
            height=350,
            margin=dict(l=40, r=40, t=70, b=40),
            title=dict(text=f"{process_name}", x=0.5, xanchor='center'),
            font=dict(family="Arial", size=12),
        )
        chart_html = fig.to_html(full_html=False)
        html_parts.append(
            f"<div style='margin-bottom:30px; border:1px solid #ddd; padding:10px;'>{chart_html}</div>"
        )

    return "".join(html_parts)

# ------------------- Forecast (Tab 3) -------------------
@app.route("/generate_forecast_table")
def generate_forecast_table():
    global forecast_demand_df

    if not app1_capacity_result:
        return "<p style='color:red;'>Please run Capacity Planner (Tab 1) first!</p>"

    if forecast_demand_df.empty:
        return "<p style='color:red;'>No monthly forecast data. Please upload a forecast spreadsheet first.</p>"

    rooms = app1_capacity_result.get("rooms", 1)
    total_capacity = app1_capacity_result.get("overall_capacity", 0)

    month_demand_map = {}
    for _, row in forecast_demand_df.iterrows():
        month = str(row["Month"]).strip()
        demand = float(row["Demand"])
        month_demand_map[month] = demand

    all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def build_forecast_table(table_title, capacity):
        demand_cells = []
        cap_cells = []
        over_cells = []

        for m in all_months:
            d_val = month_demand_map.get(m, 0)
            surplus = capacity - d_val
            color = "#d4edda" if surplus >= 0 else "#f8d7da"
            demand_cells.append(f"<td>{d_val}</td>")
            cap_cells.append(f"<td>{capacity}</td>")
            over_cells.append(f'<td style="background-color:{color};">{surplus}</td>')

        th_months = "".join(f"<th>{m}</th>" for m in all_months)
        thead_html = f"<tr><th></th>{th_months}</tr>"
        demand_html = f"<tr><td>DEMAND</td>{''.join(demand_cells)}</tr>"
        cap_html    = f"<tr><td>CAPACITY</td>{''.join(cap_cells)}</tr>"
        over_html   = f"<tr><td>OVER/UNDER</td>{''.join(over_cells)}</tr>"

        table_html = f"""
        <h3>{table_title}</h3>
        <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse; margin-bottom:20px;">
          <thead>{thead_html}</thead>
          <tbody>
            {demand_html}
            {cap_html}
            {over_html}
          </tbody>
        </table>
        """
        return table_html

    html_parts = []
    html_parts.append(build_forecast_table("Total Combined Forecast", total_capacity))

    if rooms > 1:
        per_cpu_capacity = total_capacity / rooms
        for cpu_index in range(1, int(rooms) + 1):
            title = f"Forecast for CPU #{cpu_index}"
            html_parts.append(build_forecast_table(title, per_cpu_capacity))

    return "".join(html_parts)

@app.route("/upload_forecast_demand", methods=["POST"])
def upload_forecast_demand():
    global forecast_demand_df

    file = request.files.get("forecastFile")
    if not file:
        return jsonify({"error": "No forecast file provided"}), 400

    path = os.path.join("uploads", file.filename)
    file.save(path)

    try:
        df = pd.read_excel(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    forecast_demand_df = df
    return jsonify({
        "message": "Forecast demand uploaded.",
        "data": df.to_dict(orient="records")
    })

# ------------------- Combined Index (Tabs, etc.) -------------------

@app.route("/")
def combined_index():
    return render_template("index.html")

# Uncomment the line below to run the app directly.
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
