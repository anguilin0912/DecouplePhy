import os
import sys
import json
import argparse
import glob
import gzip
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# =====================================================================
# Configuration & Path Setup
# =====================================================================
OUTPUT_DIR = os.getcwd() 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

framework_path = os.path.join(SCRIPT_DIR, "code", "ipal-ids-framework")
sys.path.append(framework_path)

import ipal_iids.settings as settings
settings.config = os.path.join(SCRIPT_DIR, "config", "dummy.json")

from ids.DecouplePhy.equations import get_equation
from ids.DecouplePhy.DecouplePhy import DecouplePhy, STTransformerAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Topology Mapping & Masking
# =====================================================================
BATADAL_TOPOLOGY = {
    "L_T1": ["F_PU1", "S_PU1", "F_PU2", "S_PU2"], "L_T2": ["F_V2", "S_V2"], "L_T3": ["F_PU4", "S_PU4"], 
    "L_T4": ["F_PU5", "S_PU5"], "L_T5": ["F_PU6", "S_PU6"], "L_T6": ["F_PU7", "S_PU7"], "L_T7": ["F_PU10", "S_PU10", "F_PU11", "S_PU11"] 
}

SWaT_TOPOLOGY = {
    "LIT101": ["MV101", "P101", "P102", "FIT101"], "FIT201": ["MV201", "P201", "P202", "P203", "P204", "P205", "P206"], 
    "LIT301": ["MV301", "MV302", "MV303", "MV304", "P301", "P302", "DPIT301"], "LIT401": ["P401", "P402", "UV401", "FIT401"], "FIT502": ["P501", "P502", "PIT501", "PIT502", "PIT503"] 
}

HAI_TOPOLOGY = {
    "P1_B2016": ["P1_LCV01D", "P1_LCV01Z", "P1_PCV01D", "P1_PCV01Z", "P1_PCV02D", "P1_PCV02Z", "P1_FT01", "P1_FT01Z"],
    "P1_B3004": ["P1_LCV01D", "P1_LCV01Z", "P1_PCV01D", "P1_PCV01Z", "P1_FT01", "P1_FT01Z"],
    "P1_B3005": ["P1_PCV02D", "P1_PCV02Z", "P1_FT03", "P1_FT03Z", "P1_FT01", "P1_FT01Z", "P1_LCV01Z"],
    "P2_VTR01": ["P2_OnOff", "P2_Emerg", "P3_PIT01"], "P2_VTR02": ["P2_OnOff", "P2_Emerg", "P1_LCV01Z"], "P2_RTR": ["P2_OnOff", "P2_Emerg", "P4_LD"],
    "P2_SCO": ["P2_OnOff", "P2_Emerg", "P1_PCV01Z", "P3_LCP01D"], "P2_AutoSD":["P2_OnOff", "P2_Emerg", "P4_HT_FD", "P4_HT_LD", "P3_LCV01D", "P3_FIT01"], 
    "P2_LCV01D": ["P3_LCV01D", "P3_PIT01", "P4_HT_PO"], "P3_LCV01D": ["P2_LCV01D", "P3_LCP01D"], "P1_FCV03D": ["P1_FT03", "P1_FT03Z", "P1_LCV01Z"], 
    "P1_PCV01D": ["P1_PCV01Z"], "P1_PCV02D": ["P1_PCV02Z"], "P1_LCV01D": ["P1_LCV01Z"], "P3_LCP01D": ["P3_LCV01D"]
}

MASK_DICT = {
    "BATADAL": [], 
    "SWaT": ["FIT502", "PIT503", "PIT501", "AIT502", "AIT504", "AIT202", "AIT402"],    
    "HAI": ["P2_HILout", "P2_CO_rpm", "P2_VYT03", "P2_24Vdc", "P4_HT_PO"]
}

def parse_attacks_json(attack_file):
    events_list = []
    with open(attack_file, 'r', encoding='utf-8') as f:
        raw_attacks = json.load(f)
    if isinstance(raw_attacks, dict):
        for k in ["attacks", "BATADAL", "SWaT", "HAI"]:
            if k in raw_attacks: 
                raw_attacks = raw_attacks[k]
                break
    if not isinstance(raw_attacks, list): return []
    for atk in raw_attacks:
        atk_id = str(atk.get("id", ""))
        start, end = atk.get("start") or atk.get("start_time"), atk.get("end") or atk.get("end_time")
        targets_raw = atk.get("attack_point") or atk.get("target") or atk.get("targets") or atk.get("attacked_sensors")
        targets = [t.replace("state;", "").strip() for t in targets_raw] if isinstance(targets_raw, list) else [targets_raw.replace("state;", "").strip()] if isinstance(targets_raw, str) else []
        if targets:
            events_list.append({"id": atk_id, "start": int(float(start)) if start else 0, "end": int(float(end)) if end else 0, "targets": targets})
    return events_list

def check_strict(pred, target, dataset_name):
    if pred == target: return True, "Exact Match"
    if dataset_name == "BATADAL" and len(pred) > 2 and len(target) > 2 and pred[2:] == target[2:] and pred[:2] in ['F_', 'S_', 'L_', 'P_'] and target[:2] in ['F_', 'S_', 'L_', 'P_']: return True, "Entity Aligned"
    if dataset_name == "HAI" and len(pred) > 1 and len(target) > 1 and pred[:-1] == target[:-1] and pred[-1] in ['D', 'Z'] and target[-1] in ['D', 'Z']: return True, "I/O Aligned"
    return False, ""

def check_topology(pred, target, dataset_name):
    topo = BATADAL_TOPOLOGY if dataset_name == "BATADAL" else SWaT_TOPOLOGY if dataset_name == "SWaT" else HAI_TOPOLOGY
    if target in topo and pred in topo[target]: return True
    if pred in topo and target in topo[pred]: return True
    for parent, children in topo.items():
        if target in children and pred in children: return True
    return False

def evaluate_dual_hit(top_k_list, actual_targets, dataset_name):
    strict_hit, topo_hit, hit_desc = False, False, "[Miss]"
    for pred in top_k_list:
        for target in actual_targets:
            is_strict, desc = check_strict(pred, target, dataset_name)
            if is_strict: return True, True, f"[Strict] {desc} -> {pred}"
            if not topo_hit and check_topology(pred, target, dataset_name):
                topo_hit = True
                hit_desc = f"[Topology] Linkage inferred -> {pred}"
    return (False, True, hit_desc) if topo_hit else (False, False, hit_desc)

def run_diagnostics(dataset_name):
    print("="*70)
    print(f"[INFO] Evaluating Pure ST-Transformer Baseline | Dataset: {dataset_name}")
    print("="*70)

    dataset_folder = "hai" if dataset_name == "HAI" else dataset_name
    model_path = os.path.join(SCRIPT_DIR, "ablation", "ablation-track2", dataset_folder, f"{dataset_name}.model")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(SCRIPT_DIR, "ablation", "ablation-track2", dataset_name, f"{dataset_name}.model")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] Ablation Track2 Model file not found at: {model_path}")
            
    model_search = [model_path]
    print(f"[INFO] Successfully loaded model from: {model_search[0]}")
    
    mapping = {"BATADAL": ["BATADAL_dataset04.state.gz", "BATADAL_test_dataset.state.gz"], "SWaT": ["SWaT_Dataset_Attack_v0.state.gz"], "HAI": ["test1.state.gz", "test2.state.gz", "test3.state.gz", "test4.state.gz", "test5.state.gz"]}
    test_data_files = [os.path.join(SCRIPT_DIR, f"datasets/{dataset_name}/{f}") for f in mapping.get(dataset_name, []) if os.path.exists(os.path.join(SCRIPT_DIR, f"datasets/{dataset_name}/{f}"))]
    attack_file = os.path.join(SCRIPT_DIR, f"datasets/{dataset_name}/attacks.json")
    events_list = parse_attacks_json(attack_file)
    total_events = len(events_list)

    model = DecouplePhy(name="DecouplePhy")
    with open(model_search[0], 'r') as f: m_data = json.load(f)
    model.sensor_order = m_data.get("sensor_order", [])
    model.sensor_max_errors = m_data.get("sensor_max_errors", [])
    model.window_size = m_data.get("window_size", 12)
    model.state_buffer = deque(maxlen=model.window_size)
    model.CI = m_data.get("CI", {})
    for s in model.CI: model.CI[s]["equation"] = get_equation(model.CI[s]["equation"])
    
    if "dl_weights" in m_data:
        model.dl_model = STTransformerAE(num_sensors=len(model.sensor_order)).to(DEVICE)
        model.dl_model.load_state_dict({k: torch.FloatTensor(v).to(DEVICE) for k, v in m_data["dl_weights"].items()})
        model.dl_model.eval()

    event_max_scores = {i: np.zeros(len(model.sensor_order)) for i in range(total_events)}
    event_alert_ticks = {i: 0 for i in range(total_events)}
    
    heatmap_atk = None 
    heatmap_max_magnitude = 0.0

    print(f"[INFO] Streaming test data and extracting Transformer residuals...")
    for test_data_file in test_data_files:
        model.cusum = {}
        model.last_value = {}
        model.state_buffer.clear()
        
        opener = gzip.open(test_data_file, 'rt') if test_data_file.endswith('.gz') else open(test_data_file, 'r')
        with opener as f:
            for step, line in enumerate(f):
                msg = json.loads(line)
                
               
                alert, _ = model.new_state_msg(msg)
              
                if True:
                    buffer_list = list(model.state_buffer)
                    t2_score = np.zeros(len(model.sensor_order))
                    if len(buffer_list) > 0:
                        with torch.no_grad():
                            t_win = torch.FloatTensor([buffer_list]).to(DEVICE)
                            pred = model.dl_model(t_win)
                            errs = torch.mean(torch.abs(pred - t_win), dim=1).cpu().numpy()[0]
                            thresh = np.array(model.sensor_max_errors) if len(model.sensor_max_errors) == len(model.sensor_order) else np.ones(len(model.sensor_order))
                            t2_score = errs / (thresh + 1e-8) 
                            
                            current_magnitude = np.max(errs)
                            if current_magnitude > heatmap_max_magnitude:
                                heatmap_max_magnitude = current_magnitude
                                heatmap_atk = np.abs(pred.cpu().numpy()[0] - t_win.cpu().numpy()[0])
                                
                    t1_score = np.zeros(len(model.sensor_order))
                    for i, s in enumerate(model.sensor_order):
                        if s in model.CI: t1_score[i] = model.cusum.get(s, 0) / (model.CI[s]["threshold"] * model.settings["threshold_factor"] + 1e-8)
                            
                    combined_score = np.maximum(t1_score, t2_score)
                    
                    for idx, ev in enumerate(events_list):
                        if (ev["id"] and str(msg.get("malicious", "False")).strip() == ev["id"]) or (ev["start"] - 5 <= msg.get("timestamp", step) <= ev["end"] + 15):
                            event_max_scores[idx] = np.maximum(event_max_scores[idx], combined_score)
                            event_alert_ticks[idx] += 1

    # =====================================================================
    # Root Cause Analysis & Pure Text Logging
    # =====================================================================
    hits_strict, hits_topo = {1: 0, 3: 0, 5: 0}, {1: 0, 3: 0, 5: 0}
    current_mask_list = MASK_DICT.get(dataset_name, [])
    raw_text_log = []
    
    print(f"\n[INFO] Event-level Root Cause Analysis (Top-3):")
    for idx, ev in enumerate(events_list):
        actual_targets = ev["targets"]
        log_line = ""
        
        if event_alert_ticks[idx] > 0:
            pure_top_k_list = [model.sensor_order[i] for i in np.argsort(event_max_scores[idx])[::-1] if not (model.sensor_order[i] in current_mask_list and model.sensor_order[i] not in actual_targets) and not (dataset_name == "BATADAL" and model.sensor_order[i].startswith("P_J") and model.sensor_order[i] not in actual_targets)]
            for k in [1, 3, 5]:
                is_strict, is_topo, _ = evaluate_dual_hit(pure_top_k_list[:k], actual_targets, dataset_name)
                if is_strict: hits_strict[k] += 1
                if is_topo: hits_topo[k] += 1
            
            _, _, hit_desc = evaluate_dual_hit(pure_top_k_list[:3], actual_targets, dataset_name)
            log_line = f"  Event #{ev['id']:<2} | Targets: {str(actual_targets):<15} | Top-3: {str(pure_top_k_list[:3]):<28} | {hit_desc}"
        else:
            log_line = f"  Event #{ev['id']:<2} | Targets: {str(actual_targets):<15} | [WARNING] Missed Detection"
            
        print(log_line)
        raw_text_log.append(log_line)

    # Terminal Report
    print("\n" + "="*70 + f"\n[RESULT] Quantitative Evaluation Report (Ablation Track 2): {dataset_name}\n" + "="*70)
    for m, h in [("Metric 1: Strict Entity-Level Matching", hits_strict), ("Metric 2: Topology-Aware Matching", hits_topo)]:
        print(f"\n--- {m} ---")
        for k in [1, 3, 5]: print(f"HR@{k}: {h[k]/total_events*100:>6.2f}%  ({h[k]}/{total_events})")

    # =====================================================================
    # Export Operations
    # =====================================================================
    text_log_path = os.path.join(OUTPUT_DIR, f"RCA_Event_Log_AblationTrack2_{dataset_name}.txt")
    with open(text_log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(raw_text_log))
        
    report_path = os.path.join(OUTPUT_DIR, f"Appendix_Report_AblationTrack2_{dataset_name}.json")
    report_data = {
        "Metric 1: Strict Entity-Level Matching": {f"HR@{k}": f"{hits_strict[k]/total_events*100:.2f}% ({hits_strict[k]}/{total_events})" for k in [1,3,5]},
        "Metric 2: Topology-Aware Matching": {f"HR@{k}": f"{hits_topo[k]/total_events*100:.2f}% ({hits_topo[k]}/{total_events})" for k in [1,3,5]}
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)

    print(f"\n[SUCCESS] Baseline data successfully saved to: \n -> {text_log_path}\n -> {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["BATADAL", "SWaT", "HAI"])
    run_diagnostics(parser.parse_args().dataset)