import json
from app.model import rule_based_decision
from collections import defaultdict
from pathlib import Path


def run_local_eval(public_json_path: str):
    data = json.loads(Path(public_json_path).read_text())
    # truth mapping by (case_id, study_id)
    truth = {}
    for item in data.get("truth", []):
        truth[(item["case_id"], item["study_id"])] = bool(item["is_relevant_to_current"]) 

    # build cases from public file's format: public file has a 'cases' key in the full dataset
    if "cases" in data:
        cases = data["cases"]
    else:
        print("Public JSON contains only truth entries; local eval requires the full cases JSON. Skipping.")
        return

    preds = {}
    total = 0
    correct = 0
    incorrect = 0
    for case in cases:
        cid = case["case_id"]
        cur = case["current_study"]
        for prior in case.get("prior_studies", []):
            key = (cid, prior.get("study_id"))
            decision = rule_based_decision(cid, cur.get("study_description",""), cur.get("study_date",""), prior.get("study_id",""), prior.get("study_description",""), prior.get("study_date",""))
            preds[key] = bool(decision)
            total += 1
            if key in truth:
                if preds[key] == truth[key]:
                    correct += 1
                else:
                    incorrect += 1

    if total == 0:
        print("No priors found to evaluate.")
        return

    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
    print(f"Evaluated {total} priors — accuracy: {accuracy:.4f} (correct={correct} incorrect={incorrect})")


if __name__ == "__main__":
    import sys
    json_path = sys.argv[1] if len(sys.argv) > 1 else "relevant_priors_public.json"
    run_local_eval(json_path)
