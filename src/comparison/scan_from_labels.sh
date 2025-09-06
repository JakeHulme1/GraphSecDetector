#!/usr/bin/env bash
set -euo pipefail
OUTDIR="${1:?usage: scan_from_labels.sh OUTDIR [JOBS]}"
JOBS="${2:-1}"

SEMGREP_CONFIGS="${SEMGREP_CONFIGS:-p/python p/security-audit}"
SEMGREP_EXCLUDES="${SEMGREP_EXCLUDES:-**/migrations/** **/tests/** **/.venv/** **/venv/** **/dist/** **/build/**}"
MAX_PY_KB="${MAX_PY_KB:-200}"
CONTEXT_PARENTS="${CONTEXT_PARENTS:-2}"
KEEP_WORKDIR="${KEEP_WORKDIR:-0}"

LABELS="$OUTDIR/labels.csv"
SCANS="$OUTDIR/scans"
TMP="$OUTDIR/tmp"
mkdir -p "$SCANS" "$TMP"

for d in jq curl tar semgrep; do command -v "$d" >/dev/null || { echo "Missing $d"; exit 1; }; done
[[ -s "$LABELS" ]] || { echo "No $LABELS"; exit 1; }

# Build unique repo|commit list
GROUPS="$TMP/groups.txt"
tail -n +2 "$LABELS" | awk -F',' '{print $1"|"$2}' | sort -u > "$GROUPS"

scan_one() {
  local grp="$1"
  local owner_repo="${grp%|*}"
  local commit="${grp#*|}"
  local out="$SCANS/${owner_repo//\//__}__${commit}.json"
  [[ -s "$out" ]] && { echo "[skip] $owner_repo@$commit"; return 0; }

  # Collect all labelled paths for this repo@commit
  mapfile -t rels < <(awk -F',' -v OR="$owner_repo" -v CM="$commit" 'NR>1 && $1==OR && $2==CM {print $3}' "$LABELS" | sort -u)
  (( ${#rels[@]} )) || { echo '{"results":[]}' > "$out"; return 0; }

  local tb="$TMP/${owner_repo//\//__}__${commit}.tgz"
  if [[ ! -s "$tb" ]]; then
    echo "[get ] $owner_repo@$commit"
    curl -sSfL "https://codeload.github.com/$owner_repo/tar.gz/$commit" -o "$tb" || { echo '{"results":[]}' > "$out"; return 0; }
  else
    echo "[use ] $tb"
  fi

  local work="$TMP/work_${owner_repo//\//__}__${commit}"
  rm -rf "$work"; mkdir -p "$work"

  # Build extraction patterns: exact files + parent dirs
  declare -a pats=()
  for rp in "${rels[@]}"; do
    pats+=("*/$rp")
    local d="$rp"
    for ((i=0;i<CONTEXT_PARENTS;i++)); do
      d="$(dirname "$d")"; [[ "$d" == "." || "$d" == "/" ]] && break
      pats+=("*/$d/*.py" "*/$d/*/*.py")
    done
  done
  pats+=("*/requirements*.txt" "*/Pipfile*" "*/pyproject.toml" "*/setup.py" "*/manage.py" "*/app.py" "*/wsgi.py" "*/asgi.py" "*/settings.py" "*/urls.py")
  mapfile -t pats < <(printf '%s\n' "${pats[@]}" | LC_ALL=C sort -u)

  tar -xzf "$tb" --wildcards --strip-components=1 -C "$work" "${pats[@]}" 2>/dev/null || true
  # keep only .py
  find "$work" -type f ! -name '*.py' -delete || true

  # Never prune the labelled files
  declare -a keep=()
  for rp in "${rels[@]}"; do
    [[ -f "$work/$rp" ]] && keep+=("$work/$rp")
  done
  if (( ${#keep[@]} )); then
    # build -not -path args
    args=( -type f -name '*.py' -size +"${MAX_PY_KB}"k )
    for k in "${keep[@]}"; do args+=( -not -path "$k" ); done
    find "$work" "${args[@]}" -delete || true
  else
    find "$work" -type f -name '*.py' -size +"${MAX_PY_KB}"k -delete || true
  fi

  if ! find "$work" -type f -name '*.py' | grep -q .; then
    echo '{"results":[]}' > "$out"
    (( KEEP_WORKDIR==1 )) || rm -rf "$work"
    return 0
  fi

  # Build Semgrep args
  args=(--json --metrics=off --timeout=0 -q -o "$out" --include '**/*.py')
  for cfg in $SEMGREP_CONFIGS; do args+=(--config "$cfg"); done
  for ex in $SEMGREP_EXCLUDES; do args+=(--exclude "$ex"); done

  echo "[scan] $owner_repo@$commit"
  semgrep "${args[@]}" "$work" || echo "[warn] semgrep non-zero"
  (( KEEP_WORKDIR==1 )) || rm -rf "$work"
}
export -f scan_one
export LABELS SCANS TMP SEMGREP_CONFIGS SEMGREP_EXCLUDES MAX_PY_KB CONTEXT_PARENTS KEEP_WORKDIR

xargs -a "$GROUPS" -n1 -P "$JOBS" bash -lc 'scan_one "$0"'
