#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash src/comparison/run_semgrep_50.sh <label | extracted_pairs/<label>> [MAX_FILES=50]

LABEL_OR_PATH="${1:?label like plain_sql OR path like extracted_pairs/plain_sql}"
MAX_FILES="${2:-50}"

need(){ command -v "$1" >/dev/null || { echo "Missing $1"; exit 1; }; }
need jq; need semgrep

# resolve label + dirs 
if [[ -d "$LABEL_OR_PATH" ]]; then
  LABEL="$(basename "$LABEL_OR_PATH")"
  PAIR_DIR="$(realpath "$LABEL_OR_PATH")"
else
  LABEL="$LABEL_OR_PATH"
  PAIR_DIR="$(realpath "extracted_pairs/$LABEL")"
fi
OUT_DIR="semgrep_results/$LABEL"
mkdir -p "$OUT_DIR"

# normalize: ensure *.py next to each *.meta.json
normalize() {
  local root="$1"
  shopt -s nullglob
  local fixed=0 missing=0
  for meta in "$root"/{VULN,FIXED}/*.meta.json; do
    [[ -f "$meta" ]] || continue
    local stem="${meta%.meta.json}"
    local code=""
    if   [[ -f "$stem.py" ]]; then code="$stem.py"
    elif [[ -f "$stem"    ]]; then code="$stem"
    else
      local cand; cand=$(ls -1 "$stem".* 2>/dev/null | grep -v '\.meta\.json$' | head -n 1 || true)
      [[ -n "${cand:-}" ]] && code="$cand"
    fi
    if [[ -z "$code" ]]; then ((missing++)); continue; fi
    local target="$stem.py"
    [[ "$code" == "$target" ]] && continue
    if [[ "$code" == *.py.py ]]; then mv -f "$code" "$target"; ((fixed++)); continue; fi
    mv -f "$code" "$target"; ((fixed++))
  done
  echo "[normalize] $LABEL fixed=$fixed missing=$missing"
}
normalize "$PAIR_DIR"

list_sorted() { find "$1" -type f -name '*.py' | LC_ALL=C sort; }
mapfile -t VULN_FILES < <(list_sorted "$PAIR_DIR/VULN" 2>/dev/null || true)
mapfile -t FIX_FILES  < <(list_sorted "$PAIR_DIR/FIXED" 2>/dev/null || true)

half=$(( MAX_FILES / 2 ))
take_v=$(( ${#VULN_FILES[@]} < half ? ${#VULN_FILES[@]} : half ))
take_f=$(( ${#FIX_FILES[@]}  < half ? ${#FIX_FILES[@]}  : half ))
remain=$(( MAX_FILES - take_v - take_f ))
if (( remain > 0 )); then
  avail_v=$(( ${#VULN_FILES[@]} - take_v ))
  avail_f=$(( ${#FIX_FILES[@]}  - take_f ))
  fill_v=$(( remain < avail_v ? remain : avail_v ))
  take_v=$(( take_v + fill_v ))
  remain=$(( remain - fill_v ))
  fill_f=$(( remain < avail_f ? remain : avail_f ))
  take_f=$(( take_f + fill_f ))
fi

TARGETS="/tmp/targets_${LABEL}_$$.txt"
: > "$TARGETS"
printf "%s\n" "${VULN_FILES[@]:0:take_v}" >> "$TARGETS"
printf "%s\n" "${FIX_FILES[@]:0:take_f}"  >> "$TARGETS"
sort -u "$TARGETS" -o "$TARGETS"

COUNT=$(wc -l < "$TARGETS" | tr -d ' ')
POS_CNT=$(( take_v ))
NEG_CNT=$(( take_f ))
echo "[select] $LABEL -> $COUNT files (VULN=$POS_CNT FIXED=$NEG_CNT cap=$MAX_FILES)"

RESULTS="$OUT_DIR/results.json"
if [[ "$COUNT" -eq 0 ]]; then
  echo "[!] No .py files to scan under $PAIR_DIR"
  echo '{"results":[]}' > "$RESULTS"
fi

run_semgrep_once() {
  local use_pro="${1:-0}"; shift || true
  local -a files=("$@")
  local -a packs_opts=()
  for P in "${PACKS[@]}"; do packs_opts+=(-c "$P"); done
  local -a opts=(scan --metrics=off --no-git-ignore -j "$(nproc)" --timeout 120 --json -o "$RESULTS")
  if [[ "$use_pro" == "1" ]]; then opts+=(--pro); fi
  env -i PATH="$PATH" HOME="$HOME" semgrep "${opts[@]}" "${packs_opts[@]}" "${files[@]}"
  return $?
}

if [[ "$COUNT" -gt 0 ]]; then
  mapfile -t FILES < "$TARGETS"
  declare -a PACKS
  if [[ -n "${SEMGREP_PACKS:-}" ]]; then
    read -r -a PACKS <<<"$SEMGREP_PACKS"
  else
    if [[ "$LABEL" =~ (sql|sqli) ]]; then
      PACKS=(p/sql-injection)  
    else
      PACKS=(p/python p/security-audit)
    fi
  fi

  set +e
  rc=0
  if [[ "${SEMGREP_PRO:-0}" == "1" ]]; then
    run_semgrep_once 1 "${FILES[@]}"; rc=$?
    if [ "$rc" -ge 2 ]; then
      echo "[fallback] Pro run failed (rc=$rc). Re-running without --pro…"
      run_semgrep_once 0 "${FILES[@]}"; rc=$?
    fi
  else
    run_semgrep_once 0 "${FILES[@]}"; rc=$?
  fi
  set -e
  if [ "$rc" -ge 2 ]; then
    echo "[warn] semgrep exited with $rc. Using whatever was written to $RESULTS."
  fi
else
  echo '{"results":[]}' > "$RESULTS"
fi

# label-aware prediction filter 
FILTER_RE=''; RULE_ID_RE=''
case "$LABEL" in
  *command*) FILTER_RE='(subprocess|os\.system|os\.popen|shell\s*=\s*true|/bin/(sh|bash)|pty\.spawn|commands\.get)';;
  *sql*|*sqli*) RULE_ID_RE='sql[-_]?injection';; 
  *xss*) FILTER_RE='(<script|javascript:|on[a-z]+\s*=|mark_safe|escape\s*=\s*False|autoescape\s*(False|off)|\|safe)';;
  *xsrf*|*csrf*) FILTER_RE='(@csrf_exempt|csrf_exempt|WTF_CSRF_ENABLED\s*=\s*False|CSRF)';;
  *open_redirect*) FILTER_RE='(redirect\(|HttpResponse(Permanent)?Redirect|url_for\(|request\.args.*(next|redirect|url))';;
  *path_disclosure*) FILTER_RE='(Traceback|DEBUG\s*=\s*True|werkzeug\.debug|Exception:|File\s*")';;
  *remote_code_execution*) FILTER_RE='(eval\(|exec\(|execfile\(|pickle\.loads|marshal\.loads|yaml\.load\(|importlib\.import_module)';;
  *) FILTER_RE='';;
esac

ABS_DIR="$(realpath "$PAIR_DIR")"
tmpdir="$(mktemp -d)"
METRICS="$OUT_DIR/metrics.json"

# Ground truth for selected set
awk '{print}' "$TARGETS" | xargs -r realpath -m | sort -u > "$tmpdir/allow.txt"
find "$ABS_DIR" -type f -name '*.meta.json' -print0 \
| while IFS= read -r -d '' m; do
    f="${m%.meta.json}.py"; [[ -f "$f" ]] || continue
    s=$(jq -r '.state' "$m"); y=0; [[ "$s" == "vulnerable" ]] && y=1
    printf "%s\t%d\n" "$(realpath "$f")" "$y"
  done | sort -u > "$tmpdir/gt_all.tsv"
awk -F'\t' -v allow="$tmpdir/allow.txt" '
  BEGIN{ while ((getline l < allow) > 0) a[l]=1; close(allow) }
  { if ($1 in a) print $0 }
' "$tmpdir/gt_all.tsv" > "$tmpdir/gt.tsv"

# Predicted positives 
PRED_PATHS="$tmpdir/pred.txt"
if [[ -s "$RESULTS" ]]; then
  if [[ -n "$RULE_ID_RE" ]]; then
    jq -r --arg idre "$RULE_ID_RE" '.results[] | select(.check_id | test($idre; "i")) | .path' "$RESULTS"
  elif [[ -n "$FILTER_RE" ]]; then
    jq -r --arg re "$FILTER_RE" '.results[] | select((.check_id | test($re; "i")) or ((.extra.lines // "") | test($re; "i"))) | .path' "$RESULTS"
  else
    jq -r '.results[].path' "$RESULTS"
  fi \
  | awk -v R="$ABS_DIR" '{ if ($0 ~ /^\//) print $0; else print R "/" $0 }' \
  | xargs -r realpath -m \
  | sort -u > "$PRED_PATHS"
else
  : > "$PRED_PATHS"
fi

# explain table
if [[ "${EXPLAIN:-0}" == "1" && -s "$RESULTS" ]]; then
  OUT_EXPLAIN="$OUT_DIR/explain.tsv"
  awk '{print $0}' "$PRED_PATHS" > "$tmpdir/pred_set.txt"
  awk -F'\t' -v predfile="$tmpdir/pred_set.txt" '
    BEGIN{ while ((getline p < predfile) > 0) pred[p]=1; close(predfile) }
    { path=$1; y=$2+0; yhat=(path in pred)?1:0; print path "\t" y "\t" yhat }
  ' "$tmpdir/gt.tsv" > "$tmpdir/base_explain.tsv"

  # collect rule ids under same filtering used for predictions
  if [[ -n "$RULE_ID_RE" ]]; then
    jq -r --arg idre "$RULE_ID_RE" '.results[] | select(.check_id | test($idre; "i")) | [.path, .check_id] | @tsv' "$RESULTS" > "$tmpdir/file_rules.tsv"
  elif [[ -n "$FILTER_RE" ]]; then
    jq -r --arg re "$FILTER_RE" '.results[] | select((.check_id | test($re; "i")) or ((.extra.lines // "") | test($re; "i"))) | [.path, .check_id] | @tsv' "$RESULTS" > "$tmpdir/file_rules.tsv"
  else
    jq -r '.results[] | [.path, .check_id] | @tsv' "$RESULTS" > "$tmpdir/file_rules.tsv"
  fi

  awk -F'\t' '{arr[$1]= (arr[$1] ? arr[$1] "," $2 : $2)} END{for (k in arr) print k "\t" arr[k]}' \
    "$tmpdir/file_rules.tsv" | sort -u > "$tmpdir/agg_rules.tsv"

  awk -F'\t' '
    FNR==NR {rules[$1]=$2; next}
    {r=(($1 in rules)?rules[$1]:""); print $1 "\t" $2 "\t" $3 "\t" r}
  ' "$tmpdir/agg_rules.tsv" "$tmpdir/base_explain.tsv" \
  | (echo -e "path\ty\tyhat\trules"; cat -) > "$OUT_EXPLAIN"
fi

# Top rules JSON
TOP_JSON="$tmpdir/top_rules.json"
if [[ -s "$RESULTS" ]]; then
  if [[ -n "$RULE_ID_RE" ]]; then
    jq -r --arg idre "$RULE_ID_RE" '.results[] | select(.check_id | test($idre; "i")) | .check_id' "$RESULTS"
  elif [[ -n "$FILTER_RE" ]]; then
    jq -r --arg re "$FILTER_RE" '.results[] | select((.check_id | test($re; "i")) or ((.extra.lines // "") | test($re; "i"))) | .check_id' "$RESULTS"
  else
    jq -r '.results[].check_id' "$RESULTS"
  fi | sort | uniq -c | sort -nr | head -n 10 \
     | awk '{printf "{\"id\":\"%s\",\"count\":%s}\n",$2,$1}' \
     | jq -s '.' > "$TOP_JSON"
else
  echo "[]" > "$TOP_JSON"
fi

# Metrics
read -r TP FP FN TN PREC REC F1 <<<"$(
  awk -F'\t' -v predfile="$PRED_PATHS" '
    BEGIN{ while ((getline p < predfile) > 0) pred[p]=1; close(predfile) }
    { path=$1; y=$2+0; yhat=(path in pred)?1:0
      if (y==1 && yhat==1) tp++;
      else if (y==0 && yhat==1) fp++;
      else if (y==1 && yhat==0) fn++;
      else tn++;
    }
    END{
      prec=(tp+fp>0)? tp/(tp+fp):0
      rec =(tp+fn>0)? tp/(tp+fn):0
      f1  =(prec+rec>0)? 2*prec*rec/(prec+rec):0
      printf "%d %d %d %d %.6f %.6f %.6f", tp,fp,fn,tn,prec,rec,f1
    }
  ' "$tmpdir/gt.tsv"
)"

POS=$(awk '$2==1' "$tmpdir/gt.tsv" | wc -l | tr -d ' ')
NEG=$(awk '$2==0' "$tmpdir/gt.tsv" | wc -l | tr -d ' ')
N=$((POS+NEG))

jq -n \
  --arg label "$LABEL" \
  --argjson n "$N" --argjson pos "$POS" --argjson neg "$NEG" \
  --argjson tp "$TP" --argjson fp "$FP" --argjson fn "$FN" --argjson tn "$TN" \
  --argjson precision "$PREC" --argjson recall "$REC" --argjson f1 "$F1" \
  --slurpfile top "$TOP_JSON" \
  '{label:$label,n:$n,positives:$pos,negatives:$neg,
    tp:$tp,fp:$fp,fn:$fn,tn:$tn,
    precision:$precision,recall:$recall,f1:$f1,
    top_rules: $top[0]}' > "$METRICS"

printf "[metrics] %s  N=%d (VULN=%d FIXED=%d)  TP=%d FP=%d FN=%d TN=%d  P=%.3f R=%.3f F1=%.3f\n" \
  "$LABEL" "$N" "$POS" "$NEG" "$TP" "$FP" "$FN" "$TN" "$PREC" "$REC" "$F1"

echo "[done] $LABEL -> $RESULTS  and  $METRICS"
[[ "${EXPLAIN:-0}" == "1" && -f "$OUT_EXPLAIN" ]] && echo "[explain] $OUT_EXPLAIN"
