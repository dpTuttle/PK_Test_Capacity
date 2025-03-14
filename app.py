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

def generate_heatmap(df, value_col, timescale=None):
    if df.empty:
        return ""
    pivot_df = df.set_index("Process")[[value_col]]
    bottleneck_value = pivot_df[value_col].min()
    threshold = bottleneck_value * 1.1
    constrained_data = pivot_df.where(pivot_df[value_col] <= threshold)
    plt.figure(figsize=(5, 3))
    ax = sns.heatmap(pivot_df, annot=True, cmap="Greys", fmt=".1f", cbar=False)
    sns.heatmap(constrained_data, annot=False, cmap="Reds", fmt=".1f", 
                cbar_kws={"label": "Capacity"}, ax=ax, linewidths=3, linecolor='black')
    title = f"Bottleneck Heatmap ({timescale})" if timescale else "Bottleneck Heatmap"
    for process, value in pivot_df[value_col].items():
        if value == bottleneck_value:
            ax.text(0.5, list(pivot_df.index).index(process) + 0.5, "          ← Bottleneck", 
                    color="blue", fontsize=10, va="center", ha="left")
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def compute_capacity(row):
    hours_per_month = 720  # Fixed example: 30 days * 24 hours.
    labor_capacity = (row["Labor Headcount"] / row["Cycle Time (hr/unit)"]) * hours_per_month
    return labor_capacity

# ------------------- Process Class -------------------

class Process:
    """
    Each process has: name, cycle_time, labor, bsc, incubator, and process_days.
    The cycle_time is the time operators are actively working,
    while process_days is additional hold time (in days) during which incubators remain occupied.
    For incubator capacity calculations, we use an effective cycle time:
      effective_cycle_time = cycle_time + (process_days * 24)
    """
    def __init__(self, name, cycle_time, labor, bsc, incubator, process_days=0):
        self.name = name
        self.cycle_time = float(cycle_time)
        self.labor = float(labor)
        self.bsc = float(bsc)
        self.incubator = float(incubator)
        self.process_days = float(process_days)

    def capacity_per_period(self, hours_per_period, max_bsc, max_incubators, days_per_period, rooms):
        # Use original cycle time for labor and BSC constraints.
        labor_cap = (self.labor / self.cycle_time) * hours_per_period
        if self.bsc > 0:
            bsc_cap = (max_bsc / self.bsc) * (hours_per_period / self.cycle_time)
        else:
            bsc_cap = 1e9
        # Use effective cycle time for incubator constraints.
        if self.incubator > 0:
            effective_cycle_time = self.cycle_time + (self.process_days * 24)
            inc_cap = (max_incubators / self.incubator) * (hours_per_period / effective_cycle_time)
        else:
            inc_cap = 1e9
        room_cap = days_per_period * rooms
        return min(labor_cap, bsc_cap, inc_cap, room_cap)

# ------------------- Tab 1 (App1) Routes -------------------

@app.route("/upload_excel", methods=["POST"])
def upload_excel():
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
    process_data = []
    # Use explicit check for the optional columns.
    for _, row in df.iterrows():
        dept = row["Department"] if "Department" in row and pd.notnull(row["Department"]) else "MFG"
        process_days = row["Process Days"] if "Process Days" in row and pd.notnull(row["Process Days"]) else 0
        process_data.append({
            "name": row["Process"],
            "cycle_time": row["Cycle Time (hr/unit)"],
            "labor": row["Labor Headcount"],
            "bsc": row["BSC"],
            "incubator": row["Incubator"],
            "department": dept,
            "process_days": process_days
        })
    uploaded_process_df = df
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
    mfg_headcount = float(req.get("mfg_headcount", 0))
    qc_headcount = float(req.get("qc_headcount", 0))
    qa_headcount = float(req.get("qa_headcount", 0))
    process_list = req.get("processes", [])
    processes = []
    for p in process_list:
        try:
            proc = Process(
                p["name"],
                p["cycle_time"],
                p["labor"],
                p["bsc"],
                p["incubator"],
                p.get("process_days", 0)
            )
            proc.department = p.get("department", "MFG")
            processes.append(proc)
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
                "Capacity": round(cap, 1),
                "Department": proc.department
            })
        if not capacities:
            return jsonify({"message": "No processes defined"}), 200
        hard_capacity = min(c["Capacity"] for c in capacities)
        bottleneck = [c for c in capacities if c["Capacity"] == hard_capacity][0]["Process"]
        df_cap = pd.DataFrame(capacities)
        heatmap_img = generate_heatmap(df_cap, "Capacity", timescale)
        # Departmental Labor Calculation
        departments = {}
        for proc in processes:
            dept = proc.department.upper()
            if dept not in departments:
                departments[dept] = []
            departments[dept].append(proc)
        scaling_factors = []
        labor_details = {}
        for dept, proc_list in departments.items():
            required_hours = sum(hard_capacity * proc.cycle_time for proc in proc_list)
            if dept == "MFG":
                available_hours = mfg_headcount * hours_per_shift * days_per_period
            elif dept == "QC":
                available_hours = qc_headcount * hours_per_shift * days_per_period
            elif dept == "QA":
                available_hours = qa_headcount * hours_per_shift * days_per_period
            else:
                available_hours = 0
            factor = available_hours / required_hours if required_hours > 0 else 1
            if not factor or factor == 0:
                factor = 1
            scaling_factors.append(factor)
            labor_details[dept] = {
                "required_hours": round(required_hours, 2),
                "available_hours": round(available_hours, 2),
                "scaling_factor": round(factor, 2)
            }
        overall_scaling_factor = min(scaling_factors) if scaling_factors else 1
        theoretical_capacity = hard_capacity * overall_scaling_factor
        msg = (f"Max {timescale} throughput is {hard_capacity:.1f} units, with {rooms} suite(s), "
               f"{max_bsc} BSC(s), {max_incubators} Incubator(s). Bottleneck: {bottleneck}.")
        msg += (f" Based on headcount (MFG: {mfg_headcount}, QC: {qc_headcount}, QA: {qa_headcount}), "
                f"the theoretical maximum throughput is {theoretical_capacity:.1f} units.")
        app1_capacity_result = {
            "rooms": rooms,
            "max_bsc": max_bsc,
            "max_incubators": max_incubators,
            "timescale": timescale,
            "overall_capacity": hard_capacity,
            "details": capacities
        }
        return jsonify({
            "scenario": scenario,
            "capacities": capacities,
            "message": msg,
            "heatmap": heatmap_img,
            "hard_capacity": hard_capacity,
            "adjusted_throughput": theoretical_capacity,
            "labor_by_department": labor_details
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
                needed_bsc    = max(proc.bsc, proc.bsc * factor)
                needed_inc    = max(proc.incubator, proc.incubator * factor)
            results.append({
                "Process": proc.name,
                "Department": proc.department,
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
        total_needed_labor = sum(item["Needed Labor"] for item in results)
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
            "csv_data": csv_str,
            "total_needed_labor": total_needed_labor
        })

    return jsonify({"error": "Invalid scenario"}), 400

@app.route("/download_csv", methods=["POST"])
def download_csv():
    csv_data = request.form.get("csv_data", "")
    buf = io.BytesIO(csv_data.encode("utf-8"))
    return send_file(buf, as_attachment=True, download_name="ideal_setup.csv", mimetype="text/csv")

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

@app.route("/visuals", methods=["POST"])
def visuals():
    if not app1_capacity_result:
        return jsonify({"error": "Please run Capacity Planner (Tab 1) first!"}), 400
    req_data = request.json or {}
    desired_patients = float(req_data.get("desired_patients", 0))
    details = app1_capacity_result.get("details", [])
    if not details:
        return jsonify({"error": "No process capacity data available."}), 400
    charts = []
    for proc in details:
        proc_name = proc.get("Process", "Unknown")
        proc_capacity = proc.get("Capacity", 0)
        max_range = max(proc_capacity, desired_patients) * 1.2
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=proc_capacity,
            delta={
                'reference': desired_patients, 
                'increasing': {'color': "green"},
                'decreasing': {'color': "red"}
            },
            title={"text": proc_name},
            gauge={
                'axis': {'range': [None, max_range]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, max_range * 0.5], 'color': '#c7e9c0'},
                    {'range': [max_range * 0.5, max_range], 'color': '#ffffbf'}
                ],
            }
        ))
        fig.update_layout(
            margin={"t": 50, "b": 0, "l": 0, "r": 0},
            annotations=[
                {
                    "text": f"Capacity: {proc_capacity}, Demand: {desired_patients:.1f}",
                    "x": 0.5,
                    "y": 0,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 12, "color": "black"}
                }
            ]
        )
        charts.append(fig.to_plotly_json())
    import json
    return app.response_class(json.dumps(charts), mimetype='application/json')

@app.route("/generate_forecast_table", methods=["POST"])
def generate_forecast_table():
    global forecast_demand_df
    if not app1_capacity_result:
        return "<p style='color:red;'>Please run Capacity Planner (Tab 1) first!</p>"
    if forecast_demand_df.empty:
        return "<p style='color:red;'>No monthly forecast data. Please upload a forecast spreadsheet first.</p>"
    req_data = request.json or {}
    constraints = req_data.get("constraints", {})
    rooms = app1_capacity_result.get("rooms", 1)
    total_capacity = app1_capacity_result.get("overall_capacity", 0)
    month_demand_map = {}
    for _, row in forecast_demand_df.iterrows():
        month = str(row["Month"]).strip()
        demand = float(row["Demand"])
        month_demand_map[month] = demand
    # Use the months provided by the Excel instead of a fixed list:
    all_months = list(month_demand_map.keys())
    def build_forecast_table(table_title, capacity):
        demand_cells = []
        cap_cells = []
        over_cells = []
        for m in all_months:
            d_val = month_demand_map.get(m, 0)
            capacity_reduced = capacity - constraints.get(m, 0)
            if capacity_reduced < 0:
                capacity_reduced = 0
            surplus = capacity_reduced - d_val
            color = "#d4edda" if surplus >= 0 else "#f8d7da"
            demand_cells.append(f"<td>{int(round(d_val))}</td>")
            cap_cells.append(f"<td>{int(round(capacity_reduced))}</td>")
            over_cells.append(f'<td style="background-color:{color};">{int(round(surplus))}</td>')
        th_months = "".join(f"<th>{m}</th>" for m in all_months)
        thead_html = f"<tr><th></th>{th_months}</tr>"
        demand_html = f"<tr><td>DEMAND</td>{''.join(demand_cells)}</tr>"
        cap_html = f"<tr><td>CAPACITY</td>{''.join(cap_cells)}</tr>"
        over_html = f"<tr><td>OVER/UNDER</td>{''.join(over_cells)}</tr>"
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
            title = f"Forecast for Suite #{cpu_index}"
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

@app.route("/download_template/<template_type>", methods=["GET"])
def download_template(template_type):
    import io
    if template_type == "process":
        df = pd.DataFrame(columns=[
            "Process",
            "Cycle Time (hr/unit)",
            "Labor Headcount",
            "BSC",
            "Incubator",
            "Process Days",
            "Department"
        ])
        filename = "process_template.xlsx"
    elif template_type == "forecast":
        df = pd.DataFrame({
            "Month": ["" for _ in range(12)],
            "Demand": ["" for _ in range(12)]
        })
        filename = "forecast_template.xlsx"
    else:
        return "Invalid template type", 400
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return send_file(
        output,
        attachment_filename=filename,
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/download_forecast_excel", methods=["POST"])
def download_forecast_excel():
    global forecast_demand_df
    if not app1_capacity_result:
        return jsonify({"error": "Please run Capacity Planner (Tab 1) first!"}), 400
    if forecast_demand_df.empty:
        return jsonify({"error": "No monthly forecast data. Please upload a forecast spreadsheet first."}), 400
    req_data = request.json or {}
    constraints = req_data.get("constraints", {})
    rooms = app1_capacity_result.get("rooms", 1)
    total_capacity = app1_capacity_result.get("overall_capacity", 0)
    all_months = list(forecast_demand_df["Month"].dropna().astype(str))
    forecast_data = []
    for m in all_months:
        d_val = 0
        for _, row in forecast_demand_df.iterrows():
            if str(row["Month"]).strip() == m:
                d_val = float(row["Demand"])
                break
        capacity_reduced = total_capacity - constraints.get(m, 0)
        if capacity_reduced < 0:
            capacity_reduced = 0
        surplus = capacity_reduced - d_val
        forecast_data.append({
            "Month": m,
            "Demand": d_val,
            "Capacity": capacity_reduced,
            "Over/Under": surplus
        })
    df_forecast = pd.DataFrame(forecast_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_forecast.to_excel(writer, index=False, sheet_name="Combined Forecast")
        if rooms > 1:
            per_suite_capacity = total_capacity / rooms
            for cpu_index in range(1, int(rooms) + 1):
                forecast_data_cpu = []
                for m in all_months:
                    d_val = 0
                    for _, row in forecast_demand_df.iterrows():
                        if str(row["Month"]).strip() == m:
                            d_val = float(row["Demand"])
                            break
                    capacity_reduced = per_suite_capacity - constraints.get(m, 0)
                    if capacity_reduced < 0:
                        capacity_reduced = 0
                    surplus = capacity_reduced - d_val
                    forecast_data_cpu.append({
                        "Month": m,
                        "Demand": d_val,
                        "Capacity": capacity_reduced,
                        "Over/Under": surplus
                    })
                df_cpu = pd.DataFrame(forecast_data_cpu)
                sheet_name = f"Suite #{cpu_index}"
                df_cpu.to_excel(writer, index=False, sheet_name=sheet_name)
        writer.save()
    output.seek(0)
    return send_file(output,
                     as_attachment=True,
                     download_name="forecast_results.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route("/")
def combined_index():
    return render_template("index.html")

# Uncomment the following to run the app directly.
# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
