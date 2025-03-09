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

app = Flask(__name__)

# Ensure 'uploads' and 'static' directories exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('static'):
    os.makedirs('static')

class Process:
    def __init__(self, name, cycle_time, labor, equipment):
        self.name = name
        self.cycle_time = float(cycle_time)      # hours per unit
        self.labor = float(labor)               # number of people
        self.equipment = float(equipment)       # number of machines

    def capacity_per_period(self, hours_per_period, equip_throughput):
        # Labor-based capacity
        labor_capacity = (self.labor / self.cycle_time) * hours_per_period
        # Equipment-based capacity
        equipment_capacity = (self.equipment * equip_throughput) * hours_per_period
        # Return minimum
        return min(labor_capacity, equipment_capacity)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    """
    Handle the Excel file upload. We'll parse it and store the data in session
    or just return the parsed data to the frontend for display.
    """
    file = request.files.get('excelFile')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    # Save file locally in 'uploads' folder
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Read with pandas
    try:
        df = pd.read_excel(filepath, sheet_name=0)  # or specify 'Sheet1'
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # We expect columns: Process, Cycle Time (hr/unit), Labor Headcount, Equipment
    # Convert to dict
    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "name": row["Process"],
            "cycle_time": row["Cycle Time (hr/unit)"],
            "labor": row["Labor Headcount"],
            "equipment": row["Equipment"]
        })

    return jsonify({"processData": data_list})

@app.route('/calculate', methods=['POST'])
def calculate():
    """
    Main route to handle scenario 1 (max throughput) or scenario 2 (desired units),
    plus shift scheduling and time-scale selection (annual, monthly, weekly, daily).
    """
    req = request.json
    if not req:
        return jsonify({"error": "No data"}), 400

    # 1) SHIFT SCHEDULING
    shifts_per_day = float(req.get("shifts_per_day", 1))
    hours_per_shift = float(req.get("hours_per_shift", 8))

    # 2) TIME SCALE (annual, monthly, weekly, daily)
    timescale = req.get("timescale", "monthly")  # default
    days_per_period = get_days_for_timescale(timescale)

    # total hours in the chosen time scale
    hours_per_period = shifts_per_day * hours_per_shift * days_per_period

    # 3) Eqp throughput (units per machine per hour)
    equip_throughput = float(req.get("equip_throughput", 1.0))

    # 4) Gather processes
    process_list = req.get("processes", [])
    processes = []
    for p in process_list:
        proc = Process(p["name"], p["cycle_time"], p["labor"], p["equipment"])
        processes.append(proc)

    # 5) SCENARIO
    scenario = req.get("scenario")

    if scenario == "max_throughput":
        # Scenario 1: Max monthly/annual/weekly/daily throughput
        capacities = []
        for proc in processes:
            cap = proc.capacity_per_period(hours_per_period, equip_throughput)
            capacities.append({"Process": proc.name, "Capacity": round(cap, 1)})

        if not capacities:
            return jsonify({"message": "No processes defined"}), 200

        # System throughput = min of capacities
        min_cap = min([c["Capacity"] for c in capacities])
        bottleneck = [c for c in capacities if c["Capacity"] == min_cap][0]["Process"]

        # generate heatmap
        df_cap = pd.DataFrame(capacities)
        heatmap_img = generate_heatmap(df_cap, "Capacity")

        msg = f"Based on current conditions, your max {timescale} throughput is {min_cap:.1f} units. The bottleneck is {bottleneck}."
        return jsonify({
            "scenario": scenario,
            "capacities": capacities,
            "message": msg,
            "heatmap": heatmap_img
        })

    elif scenario == "desired_units":
        # Scenario 2: Achieve a user-defined production goal
        desired_units = float(req.get("desired_units", 1000))

        results = []
        for proc in processes:
            current_cap = proc.capacity_per_period(hours_per_period, equip_throughput)

            if current_cap >= desired_units:
                # No changes needed
                note = "No changes"
                needed_labor = proc.labor
                needed_equipment = proc.equipment
            else:
                # Solve for extra labor or equipment
                # We'll do a naive approach:
                # If equipment is fixed, how much labor needed?
                #   desired_units <= (labor/cycle_time)*hours_per_period
                #   labor >= desired_units * cycle_time / hours_per_period
                needed_labor_calc = (desired_units * proc.cycle_time) / hours_per_period

                # Also check if equipment is a bottleneck:
                #   desired_units <= equipment * equip_throughput * hours_per_period
                #   needed_equipment >= desired_units / (equip_throughput * hours_per_period)
                needed_equipment_calc = math.ceil(desired_units / (equip_throughput * hours_per_period))

                # We'll pick whichever is bigger relative to current. (Or you can pick a strategy.)
                # For example:
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

        # Build CSV data for download
        csv_str = build_csv(results)

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

# ----- Helper functions -----

def get_days_for_timescale(timescale):
    """Return how many days define 'one period' based on timescale."""
    if timescale == "annual":
        return 365  # or 250 if business days
    elif timescale == "monthly":
        return 30   # or 22 business days
    elif timescale == "weekly":
        return 7    # or 5 business days
    elif timescale == "daily":
        return 1
    return 30  # default monthly

def generate_heatmap(df, value_col):
    """Generate a heatmap from a DataFrame with columns: [Process, value_col]."""
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
    """Convert a list of dicts to CSV string."""
    if not results:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
    return output.getvalue()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
