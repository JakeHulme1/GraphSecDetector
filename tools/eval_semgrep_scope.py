#!/usr/bin/env python3
import argparse, csv, json, os, sys, re
from collections import Counter, defaultdict
from pathlib import PurePosixPath as P

def _norm_check_id(raw: str) -> str:
    s = str(raw or "").strip().lower()
    s = re.split(r'[:/\\]', s)[-1]
    parts = s.split(".")
    if len(parts) > 1:
        s = parts[-1]
    return s

# Built-in heuristics: substrings to look for in Semgrep check_id for each family.
DEFAULT_FAMILY_RULE_SUBSTR = {
    "plain_command_injection": [
        "subprocess-shell-true", "os-system", "popen", "shell", "command-injection"
    ],
    "plain_remote_code_execution": [
        "exec-detected", "eval", "pickle.avoid-pickle", "yaml.load", "code-injection"
    ],
    "plain_xss": [
        "avoid-mark-safe", "jinja", "autoescape", "xss"
    ],
    "plain_open_redirect": [
        "open-redirect", "flask-url-for-external-true", "redirect"
    ],
    "plain_sql": [
        "sql-injection", "raw-sql", "sqlalchemy", "cursor.execute", "execute"
    ],
    "plain_xsrf": [
        "csrf", "csrf-exempt", "csrf-disabled"
    ],
}

def load_json_results(path: str):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        # Semgrep JSON should have .results list
        res = data.get("results", [])
        if not isinstance(res, list):
            return []
        return res
    except Exception:
        return []

def load_family_map(path: str | None):
    if not path:
        return DEFAULT_FAMILY_RULE_SUBSTR
    try:
        with open(path, "r") as f:
            override = json.load(f)
        base = DEFAULT_FAMILY_RULE_SUBSTR.copy()
        base.update({k: [s.lower() for s in v] for k, v in override.items()})
        return base
    except Exception:
        return DEFAULT_FAMILY_RULE_SUBSTR

def path_in_dir_scope(hit_path: str, labelled_rel: str, depth: int) -> bool:
    lp = P(labelled_rel)
    if str(hit_path).endswith(labelled_rel):
        return True
    anc = lp.parent
    for _ in range(max(0, depth - 1)):
        anc = anc.parent
    prefix = "" if str(anc) in ("", ".") else str(anc).rstrip("/") + "/"
    return str(hit_path).startswith(prefix)

def prf(c: Counter):
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F

def total_n(c: Counter) -> int:
    return c["tp"] + c["fp"] + c["fn"] + c["tn"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory with labels.csv and scans/")
    ap.add_argument("--scope", default="file", choices=["file", "dir", "repo_family"])
    ap.add_argument("--dir-depth", type=int, default=1, help="Ancestor depth for dir scope (1=same dir)")
    ap.add_argument("--family-map", default=None, help="Optional JSON file mapping family -> [rule substrings]")
    args = ap.parse_args()

    outdir = args.outdir
    labels_csv = os.path.join(outdir, "labels.csv")
    scans_dir = os.path.join(outdir, "scans")
    os.makedirs(scans_dir, exist_ok=True)

    if not os.path.exists(labels_csv):
        print(f"Missing {labels_csv}", file=sys.stderr)
        sys.exit(1)

    fam_map = load_family_map(args.family_map)

    with open(labels_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    scans_cache: dict[tuple[str, str], list[dict]] = {}
    total_findings = 0
    groups = set()
    for row in rows:
        key = (row["owner/repo"], row["commit"])
        if key not in scans_cache:
            scan_path = os.path.join(scans_dir, f"{key[0].replace('/','__')}__{key[1]}.json")
            results = load_json_results(scan_path)
            scans_cache[key] = results
            total_findings += len(results)
            groups.add(key)

    overall = Counter()
    by_family = defaultdict(Counter)
    top_rules = defaultdict(Counter)

    for row in rows:
        owner_repo = row["owner/repo"]
        commit = row["commit"]
        rel = row["path"]
        fam = (row["family"] or "").strip()
        lab = (row["label"] or "").strip().lower()
        gold = lab.startswith("vuln") or lab in {"pos","1","true"}

        results = scans_cache.get((owner_repo, commit), [])
        found = False
        matched_rule_ids: list[str] = []

        if args.scope == "file":
            for r in results:
                p = r.get("path", "")
                if p.endswith(rel):
                    found = True
                    raw = r.get("check_id") or (r.get("extra") or {}).get("message") or ""
                    matched_rule_ids.append(_norm_check_id(raw))

        elif args.scope == "dir":
            for r in results:
                p = r.get("path", "")
                if path_in_dir_scope(p, rel, args.dir_depth):
                    found = True
                    raw = r.get("check_id") or (r.get("extra") or {}).get("message") or "unknown_rule"
                    matched_rule_ids.append(_norm_check_id(raw))

        else:  # repo_family
            substrs = [s.lower() for s in fam_map.get(fam, [])]
            for r in results:
                raw = r.get("check_id") or (r.get("extra") or {}).get("message") or ""
                rid = _norm_check_id(raw)
                if substrs and any((rid == s) or (s in rid) for s in substrs):
                    found = True
                    matched_rule_ids.append(rid)
            # If no mapping configured for this family, count ANY finding as a hit
            if not substrs and results and not found:
                found = True
                for r in results:
                    raw = r.get("check_id") or (r.get("extra") or {}).get("message") or ""
                    matched_rule_ids.append(_norm_check_id(raw))



        if gold and found:
            overall["tp"] += 1; by_family[fam]["tp"] += 1
            for rid in matched_rule_ids:
                top_rules[fam][rid] += 1
        elif gold and not found:
            overall["fn"] += 1; by_family[fam]["fn"] += 1
        elif (not gold) and found:
            overall["fp"] += 1; by_family[fam]["fp"] += 1
        else:
            overall["tn"] += 1; by_family[fam]["tn"] += 1

    # Print and JSON payload
    Pm, Rm, Fm = prf(overall)
    print(f"Repos scanned: {len(groups)}   Findings across all repos: {total_findings}")
    print(f"Overall: TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} TN={overall['tn']}")
    print(f"P={Pm:.3f} R={Rm:.3f} F1={Fm:.3f}\n")

    print(f"{'Family':28s}  {'N':>5s}  {'P':>6s} {'R':>6s} {'F1':>6s}")
    print("-" * 56)

    def add_up(c: Counter) -> int: return c["tp"] + c["fp"] + c["fn"] + c["tn"]

    macro_vals = []
    wnumP = wnumR = wnumF = 0.0
    wden = 0
    by_label_payload = []

    for fam in sorted(by_family.keys()):
        c = by_family[fam]
        p, r, f1 = prf(c)
        n = add_up(c)
        macro_vals.append((p, r, f1))
        wnumP += p * n; wnumR += r * n; wnumF += f1 * n; wden += n
        print(f"{fam:28s}  {n:5d}  {p:6.3f} {r:6.3f} {f1:6.3f}")
        tr = [{"id": rid, "count": cnt} for rid, cnt in top_rules[fam].most_common(3)]
        by_label_payload.append({
            "label": fam, "n": n,
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"], "tn": c["tn"],
            "precision": p, "recall": r, "f1": f1,
            "top_rules": tr
        })

    if macro_vals:
        mp = sum(x for x, _, _ in macro_vals) / len(macro_vals)
        mr = sum(y for _, y, _ in macro_vals) / len(macro_vals)
        mf = sum(z for _, _, z in macro_vals) / len(macro_vals)
    else:
        mp = mr = mf = 0.0

    wp = wnumP / wden if wden else 0.0
    wr = wnumR / wden if wden else 0.0
    wf = wnumF / wden if wden else 0.0

    payload = {
        "by_label": by_label_payload,
        "micro": {
            "tp": overall["tp"], "fp": overall["fp"], "fn": overall["fn"], "tn": overall["tn"],
            "precision": Pm, "recall": Rm, "f1": Fm
        },
        "macro": {"precision": mp, "recall": mr, "f1": mf},
        "weighted_macro": {"precision": wp, "recall": wr, "f1": wf},
    }

    out_json = os.path.join(outdir, f"summary_{args.scope}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_json}")

if __name__ == "__main__":
    main()
