#!/usr/bin/env bash
set -euo pipefail

RESULTS_ROOT="${1:-semgrep_results}"
OUT_SUMMARY="${2:-$RESULTS_ROOT/summary.json}"

need(){ command -v "$1" >/dev/null || { echo "Missing $1"; exit 1; }; }
need jq

mapfile -t METAS < <(find "$RESULTS_ROOT" -maxdepth 2 -type f -name metrics.json | sort)
[[ ${#METAS[@]} -gt 0 ]] || { echo "[!] No metrics.json found under $RESULTS_ROOT"; exit 1; }

jq -s -f /dev/stdin "${METAS[@]}" > "$OUT_SUMMARY" <<'JQ'
# Inputs: array of per-label metrics objects (tp,fp,fn,tn,precision,recall,f1,n,label,top_rules)

def f1(p; r): if (p + r) > 0 then (2 * p * r) / (p + r) else 0 end;

. as $rows
| {
    by_label: $rows,
    micro: (
      (
        reduce $rows[] as $r ({tp:0, fp:0, fn:0, tn:0};
          .tp += ($r.tp // 0)
          | .fp += ($r.fp // 0)
          | .fn += ($r.fn // 0)
          | .tn += ($r.tn // 0)
        )
      ) as $c
      | {
          tp: $c.tp, fp: $c.fp, fn: $c.fn, tn: $c.tn,
          precision: ( if ($c.tp + $c.fp) > 0 then ($c.tp / ($c.tp + $c.fp)) else 0 end ),
          recall:    ( if ($c.tp + $c.fn) > 0 then ($c.tp / ($c.tp + $c.fn)) else 0 end )
        }
      | . + { f1: f1(.precision; .recall) }
    ),
    macro: (
      {
        precision: ( ($rows | map(.precision // 0) | add) / ( ($rows | length) // 1 ) ),
        recall:    ( ($rows | map(.recall    // 0) | add) / ( ($rows | length) // 1 ) ),
        f1:        ( ($rows | map(.f1        // 0) | add) / ( ($rows | length) // 1 ) )
      }
    ),
    weighted_macro: (
      # Precompute weighted sums to avoid inline division in object literals
      ( [$rows[] | {w:(.n // 0), p:(.precision // 0), r:(.recall // 0), f:(.f1 // 0)}] ) as $m
      | ( $m | map(.w) | add ) as $W
      | ( reduce $m[] as $x ({wp:0, wr:0, wf:0};
            .wp += ($x.w * $x.p)
          | .wr += ($x.w * $x.r)
          | .wf += ($x.w * $x.f)
        )
        ) as $S
      | if ($W // 0) > 0
        then { precision: ($S.wp / $W), recall: ($S.wr / $W), f1: ($S.wf / $W) }
        else { precision: 0, recall: 0, f1: 0 }
        end
    )
  }
JQ

echo "[OK] Wrote summary -> $OUT_SUMMARY"
