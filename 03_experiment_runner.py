import argparse
import os
import concurrent.futures
import csv
import json
import logging
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import xml.etree.ElementTree as ET

# ---- CONFIG ----
MODELS = {
    "qwen2.5-3b": "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
    "llama-3.2-3b": "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "qwen2.5-7b": "./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}
BATCH_LEVELS = [1, 4, 8, 16]
REPS = 20 # Can be adjusted
PORT = 8080

SYSTEM_PROMPT = (
    "You are an IT ticket triage assistant. Given a short employee request, "
    "output an XML ticket with fields: request_type, resource, scope, "
    "severity, requester_count_estimate, justification. Infer scope and severity from context."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def build_prompt(query: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nRequest: {query}\n\nOutput:"

def get_windows_ip():
    try:
        out = subprocess.check_output("ip route list default", shell=True).decode()
        return out.split("via")[1].split()[0].strip()
    except Exception:
        return "127.0.0.1"

WINDOWS_IP = get_windows_ip()

def start_server(model_path: str, batch_size: int, is_cpu: bool = False):
    """Sends a start command to the Windows manager script."""
    model_name = os.path.basename(model_path)
    ngl = 0 if is_cpu else 99
    
    windows_model_path = f"\\\\wsl.localhost\\Debian\\home\\incor\\minor\\models\\{model_name}"
    cmd = (
        f"llama.exe serve -m {windows_model_path} "
        f"--host 0.0.0.0 --port {PORT} -c 8192 -b 2048 -np {batch_size} "
        f"--threads 8 -ngl {ngl}"
    )
    
    logging.info(f"Commanding Windows Manager to start: {cmd}")
    req = urllib.request.Request(f"http://{WINDOWS_IP}:5000/start", data=json.dumps({"cmd": cmd}).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
    except Exception as e:
        raise RuntimeError(f"Could not connect to Windows manager at {WINDOWS_IP}:5000. Is windows_manager.py running in PowerShell? Error: {e}")
    
    # Wait for health check
    start_time = time.time()
    while time.time() - start_time < 60:
        try:
            resp = urllib.request.urlopen(f"http://{WINDOWS_IP}:{PORT}/health")
            if resp.getcode() == 200:
                logging.info("Server is fully loaded and healthy on Windows!")
                class DummyProc:
                    def poll(self): return None
                return DummyProc()
        except Exception:
            pass
        time.sleep(2)
        
    stop_server(None)
    raise RuntimeError("Server failed to become healthy within 60 seconds.")

def stop_server(proc):
    """Sends a stop command to the Windows manager script."""
    req = urllib.request.Request(f"http://{WINDOWS_IP}:5000/stop", data=b'{}', headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req)
        logging.info("Commanded Windows Manager to stop server.")
        time.sleep(2) # Give Windows a moment to clean up process
    except Exception as e:
        logging.error(f"Failed to stop Windows server: {e}")

def send_request(prompt: str, grammar: str):
    """Sends a generation request to the llama-server completion endpoint."""
    url = f"http://{WINDOWS_IP}:{PORT}/completion"
    payload = {
        "prompt": prompt,
        "n_predict": 250,
        "temperature": 0.0,
        "grammar": grammar,
        "seed": 42
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    start_t = time.perf_counter()
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            elapsed = time.perf_counter() - start_t
            return result.get('content', '').strip(), elapsed
    except Exception as e:
        logging.error(f"Request failed: {e}")
        return None, time.perf_counter() - start_t

def parse_xml_to_dict(xml_string: str):
    """Extracts field values from the model's XML string output."""
    try:
        # Wrap in a root element in case it's malformed or just tags
        if not xml_string.startswith("<ticket>"):
            xml_string = f"<ticket>{xml_string}</ticket>"
        
        root = ET.fromstring(xml_string)
        parsed = {}
        for child in root:
            parsed[child.tag] = child.text.strip() if child.text else ""
        return parsed
    except ET.ParseError:
        return {}

def matches_gold(value: str, gold_list: list) -> bool:
    """Checks if the predicted value is in the accepted gold list."""
    if not value or not gold_list: return False
    return value in gold_list

def main():
    parser = argparse.ArgumentParser(description="Run IT Triage Nondeterminism Experiment")
    parser.add_argument("--models", type=str, help="Comma separated model keys to run", default="qwen2.5-3b,llama-3.2-3b,qwen2.5-7b")
    parser.add_argument("--reps", type=int, default=REPS, help="Number of repetitions per query")
    parser.add_argument("--cpu-only", action="store_true", help="Run a CPU-only control condition")
    args = parser.parse_args()

    # 1. Load Dataset
    dataset_path = Path("it_query_dataset.json")
    if not dataset_path.exists():
        dataset_path = Path("llm_context/it_query_dataset.json") # fallback
    with open(dataset_path, "r") as f:
        dataset = json.load(f)["queries"]

    # 2. Load Grammar
    grammar_path = Path("ticket_schema.gbnf")
    with open(grammar_path, "r") as f:
        grammar_str = f.read()

    selected_models = [m.strip() for m in args.models.split(',')]
    
    out_path = Path("experiment_results.csv")
    fieldnames = [
        "model", "is_cpu", "batch_condition", "use_gbnf", "query_id", "ambiguous", "rep_num",
        "raw_output", "latency_sec",
        "pred_request_type", "pred_resource", "pred_scope", "pred_severity", "pred_count", "pred_just",
        "gold_match_scope", "gold_match_severity", "gold_match_count", "exact_match"
    ]

    # Write header if new file
    if not out_path.exists():
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for model_key in selected_models:
        if model_key not in MODELS:
            logging.warning(f"Model {model_key} not in config, skipping.")
            continue
        model_path = MODELS[model_key]
        if not Path(model_path).exists():
            logging.error(f"Model file {model_path} not found. Did you download it?")
            continue

        conditions = [(b, False) for b in BATCH_LEVELS]
        if args.cpu_only:
            conditions.append((1, True)) # CPU control at batch=1

        for batch_size, is_cpu in conditions:
            logging.info(f"=== Starting Run: Model={model_key} | Batch={batch_size} | CPU={is_cpu} ===")
            
            try:
                server_proc = start_server(model_path, batch_size, is_cpu)
            except RuntimeError as e:
                logging.error(str(e))
                continue

            for use_gbnf in [True, False]:
                logging.info(f"  -> Testing condition: GBNF={use_gbnf}")
                grammar_payload = grammar_str if use_gbnf else ""
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    for q in dataset:
                        prompt = build_prompt(q["query"])
                        
                        # Submit all REPS for this query to the pool
                        futures = {
                            executor.submit(send_request, prompt, grammar_payload): rep
                            for rep in range(1, args.reps + 1)
                        }

                        for future in concurrent.futures.as_completed(futures):
                            rep = futures[future]
                            raw_output, latency = future.result()
                            
                            if raw_output is None:
                                logging.error(f"Failed query {q['id']} rep {rep}")
                                raw_output = ""
                            
                            # Parse Fields
                            parsed = parse_xml_to_dict(raw_output)
                            p_scope = parsed.get("scope", "")
                            p_sev = parsed.get("severity", "")
                            p_count = parsed.get("requester_count_estimate", "")

                            # Check Gold matches
                            match_scope = matches_gold(p_scope, q.get("gold_scope", []))
                            match_sev = matches_gold(p_sev, q.get("gold_severity", []))
                            match_count = matches_gold(p_count, q.get("gold_count_estimate", []))
                            exact_match = match_scope and match_sev and match_count
                            
                            row = {
                                "model": model_key,
                                "is_cpu": is_cpu,
                                "batch_condition": batch_size,
                                "use_gbnf": use_gbnf,
                                "query_id": q["id"],
                                "ambiguous": q.get("ambiguous", False),
                                "rep_num": rep,
                                "raw_output": raw_output,
                                "latency_sec": round(latency, 3),
                                "pred_request_type": parsed.get("request_type", ""),
                                "pred_resource": parsed.get("resource", ""),
                                "pred_scope": p_scope,
                                "pred_severity": p_sev,
                                "pred_count": p_count,
                                "pred_just": parsed.get("justification", ""),
                                "gold_match_scope": match_scope,
                                "gold_match_severity": match_sev,
                                "gold_match_count": match_count,
                                "exact_match": exact_match
                            }

                            # Append to CSV
                            with open(out_path, "a", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writerow(row)
                            
                        logging.info(f"Finished query {q['id']} (all reps)")

            stop_server(server_proc)
            logging.info(f"=== Completed Run: Model={model_key} | Batch={batch_size} | CPU={is_cpu} ===\n")

    logging.info(f"All experiments finished. Results saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()

