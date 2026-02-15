#!/usr/bin/env bash
#
# W1 (Expand Sample Size) + W2 (Self-Consistency at Matched Compute)
#
# Parallelization:
#   W1: 2 processes  (1 per dataset, each with --concurrency 30)
#   W2: 8 processes  (4 models x 2 datasets, each single-threaded)
#   Total: 10 concurrent processes -> wall time ~90 min (limited by W1 debates)
#
# Estimated cost: ~$12 total (~$6 W1 debates + ~$5.50 W2 SC)
# Wall time:      ~90 min parallelized (was ~6.5h sequential)
#
set -uo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"

LOGDIR="logs/w1_w2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
START_TIME=$(date +%s)

# Expected totals
W1_PER_DS=900     # 100 cases x 3 reps x 3 models
W2_PER_DS=23600   # 100 cases x 59 reps x 4 models
W2_PER_MODEL_DS=5900  # 100 cases x 59 reps x 1 model
TOTAL_ALL=$(( (W1_PER_DS + W2_PER_DS) * 2 ))  # 49,000 per dataset x 2

# Model lists
DEBATE_MODELS="gpt-4o-mini,gpt-4.1-mini,gemini/gemini-2.5-flash"
SC_MODEL_LIST=(gpt-4o-mini gpt-4.1-mini "gemini/gemini-2.5-flash" "xai/grok-4-1-fast-non-reasoning-latest")
DATASETS=(nlsy97 compas)

# ---- PID tracking and cleanup ----
ALL_PIDS=()
W1_PIDS=()
W2_PIDS=()
W2_PID_MAP=()   # "pid:dataset:model" entries for tracking
MONITOR_PID=""

cleanup() {
    echo ""
    echo ">>> Caught interrupt, stopping all jobs..."
    [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null
    for pid in "${ALL_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo ">>> Cleaned up. Partial results preserved (debate runner is restartable)."
    exit 1
}
trap cleanup INT TERM

safe_name() {
    echo "$1" | tr '/:.' '_'
}

upper() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

# ---- Status monitor function ----
print_status() {
    local now elapsed elapsed_h elapsed_m
    now=$(date +%s)
    elapsed=$(( now - START_TIME ))
    elapsed_h=$(( elapsed / 3600 ))
    elapsed_m=$(( (elapsed % 3600) / 60 ))

    printf "\n"
    printf "┌──────────────────────────────────────────────────────────────────────┐\n"
    printf "│ STATUS UPDATE                                   %dh %02dm elapsed │\n" "$elapsed_h" "$elapsed_m"
    printf "├───────────────┬──────────────┬───────────┬────────────────────────────┤\n"
    printf "│ Block         │ Progress     │ Accuracy  │ Detail                     │\n"
    printf "├───────────────┼──────────────┼───────────┼────────────────────────────┤\n"

    local total_done=0

    # ---- W1: Parse debate progress from logs + JSON files ----
    for ds in "${DATASETS[@]}"; do
        local log="$LOGDIR/w1_${ds}.log"
        local label="W1-$(upper "$ds")"
        local done=0 pct="--" acc_str="--" detail=""

        if [ -f "$log" ]; then
            local prog_line
            prog_line=$(grep '\[progress\]' "$log" 2>/dev/null | tail -1)
            if [ -n "$prog_line" ]; then
                done=$(echo "$prog_line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
                local of=$(echo "$prog_line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f2)
                if [ -n "$of" ] && [ "$of" -gt 0 ] 2>/dev/null; then
                    pct=$(( done * 100 / of ))%
                fi
                local eta
                eta=$(echo "$prog_line" | grep -oE 'ETA: [0-9]+m' | head -1)
                [ -n "$eta" ] && detail="$eta"
            fi

            if grep -q "All debates completed" "$log" 2>/dev/null; then
                done="$W1_PER_DS"; pct="DONE"; detail=""
            fi

            local dir
            dir=$(ls -dt transcripts_ver26_${ds}_* 2>/dev/null | head -1)
            if [ -n "$dir" ] && [ -d "$dir" ]; then
                local json_ct correct wrong scored
                json_ct=$(find "$dir" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
                if [ "$json_ct" -gt 0 ]; then
                    correct=$(grep -rl '"prediction_accurate": true' "$dir" 2>/dev/null | wc -l | tr -d ' ')
                    wrong=$(grep -rl '"prediction_accurate": false' "$dir" 2>/dev/null | wc -l | tr -d ' ')
                    scored=$(( correct + wrong ))
                    if [ "$scored" -gt 0 ]; then
                        acc_str="${correct}/${scored} $(( correct * 100 / scored ))%"
                    fi
                    done="$json_ct"
                fi
            fi
        else
            detail="not started"
        fi

        total_done=$(( total_done + done ))
        printf "│ %-13s │ %4s/%-4s %4s│ %9s │ %-26s │\n" \
            "$label" "$done" "$W1_PER_DS" "$pct" "$acc_str" "$detail"
    done

    # ---- W2: Parse SC progress per model from logs ----
    for ds in "${DATASETS[@]}"; do
        local ds_done=0 ds_models_done=0
        for model in "${SC_MODEL_LIST[@]}"; do
            local sn
            sn=$(safe_name "$model")
            local log="$LOGDIR/w2_${ds}_${sn}.log"

            if [ -f "$log" ]; then
                local prog_line mdone=0
                prog_line=$(grep 'Progress:' "$log" 2>/dev/null | tail -1)
                if [ -n "$prog_line" ]; then
                    mdone=$(echo "$prog_line" | grep -oE '[0-9]+/[0-9]+' | head -1 | cut -d/ -f1)
                fi
                if grep -q "Saved detailed results\|Done!" "$log" 2>/dev/null; then
                    mdone="$W2_PER_MODEL_DS"
                    ds_models_done=$(( ds_models_done + 1 ))
                fi
                ds_done=$(( ds_done + mdone ))
            fi
        done

        local label="W2-$(upper "$ds")"
        local pct="--" detail="${ds_models_done}/4 models"
        if [ "$ds_done" -gt 0 ] && [ "$W2_PER_DS" -gt 0 ]; then
            pct=$(( ds_done * 100 / W2_PER_DS ))%
        fi
        if [ "$ds_models_done" -eq 4 ]; then
            pct="DONE"; ds_done="$W2_PER_DS"; detail=""
        fi

        total_done=$(( total_done + ds_done ))
        printf "│ %-13s │ %4s/%-4s %4s│ %9s │ %-26s │\n" \
            "$label" "$ds_done" "$W2_PER_DS" "$pct" "--" "$detail"
    done

    printf "├───────────────┴──────────────┴───────────┴────────────────────────────┤\n"

    if [ "$total_done" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
        local remaining=$(( TOTAL_ALL - total_done ))
        local eta_sec=$(( remaining * elapsed / total_done ))
        local eta_h=$(( eta_sec / 3600 ))
        local eta_m=$(( (eta_sec % 3600) / 60 ))
        printf "│ Overall: %d/%d (%d%%)                    ETA: ~%dh %02dm remaining │\n" \
            "$total_done" "$TOTAL_ALL" "$(( total_done * 100 / TOTAL_ALL ))" "$eta_h" "$eta_m"
    else
        printf "│ Overall: %d/%d — warming up...                                    │\n" \
            "$total_done" "$TOTAL_ALL"
    fi
    printf "└──────────────────────────────────────────────────────────────────────┘\n"
}

run_monitor() {
    sleep 60
    print_status
    while true; do
        sleep 300
        print_status
    done
}

# ==============================================================
echo "============================================================"
echo "W1 + W2 Experiment Runner (fully parallelized)"
echo "Logs: $LOGDIR/"
echo "============================================================"

# ---- Step 0: Generate expanded vignettes ----
echo ""
echo ">>> Step 0: Generating 100-case vignette CSVs..."
python src/expand_vignettes.py 2>&1 | tee "$LOGDIR/step0_expand.log"
echo ">>> Vignettes ready."

# ---- Launch W1: 2 debate processes ----
echo ""
echo "============================================================"
echo "Launching W1: 2 debate processes (concurrency 30 each)"
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    python src/step1_ai-debators_ver26.py \
        --dataset "$ds" \
        --vignettes "$PROJECT_ROOT/data/${ds}_vignettes_100.csv" \
        --ensemble "$DEBATE_MODELS" \
        --cases 100 --repeats 3 --concurrency 30 \
        > "$LOGDIR/w1_${ds}.log" 2>&1 &
    pid=$!
    ALL_PIDS+=($pid)
    W1_PIDS+=($pid)
    echo "  [PID $pid] W1-$(upper "$ds") -> $LOGDIR/w1_${ds}.log"
done

# ---- Launch W2: 8 SC processes (4 models x 2 datasets) ----
echo ""
echo "============================================================"
echo "Launching W2: 8 SC processes (4 models x 2 datasets)"
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    # Each model gets its own output subdirectory to avoid CSV conflicts
    for model in "${SC_MODEL_LIST[@]}"; do
        sn=$(safe_name "$model")
        outdir="results/standardllm/sc_${ds}_${sn}"
        mkdir -p "$outdir"

        python src/standardllm_evaluation.py \
            --dataset "$ds" \
            --vignettes "$PROJECT_ROOT/data/${ds}_vignettes_100.csv" \
            --models "$model" \
            --prompts system1 \
            --cases 100 --repeats 59 --temperature 0.7 \
            --output "$PROJECT_ROOT/$outdir" \
            > "$LOGDIR/w2_${ds}_${sn}.log" 2>&1 &
        pid=$!
        ALL_PIDS+=($pid)
        W2_PIDS+=($pid)
        W2_PID_MAP+=("$pid:$ds:$sn")
        echo "  [PID $pid] W2-$(upper "$ds")/${sn} -> $LOGDIR/w2_${ds}_${sn}.log"
    done
done

echo ""
echo ">>> 10 processes launched. Status updates every 5 min."
echo ">>> Manual check: tail -f $LOGDIR/w1_nlsy97.log"
echo ""

# ---- Start background monitor ----
run_monitor &
MONITOR_PID=$!

# ---- Wait for all W2 jobs, then merge + aggregate SC ----
echo ">>> Waiting for W2 (SC) jobs..."
W2_OK=1
for pid in "${W2_PIDS[@]}"; do
    wait "$pid" || { echo "WARN: W2 process $pid failed"; W2_OK=0; }
done
echo ">>> All W2 jobs finished."

# ---- Merge per-model SC results into one CSV per dataset ----
echo ""
echo "============================================================"
echo "Merging and aggregating SC results..."
echo "============================================================"

for ds in "${DATASETS[@]}"; do
    merged="results/standardllm/sc_${ds}_merged.csv"
    header_written=0

    for model in "${SC_MODEL_LIST[@]}"; do
        sn=$(safe_name "$model")
        outdir="results/standardllm/sc_${ds}_${sn}"
        # Find the results CSV (timestamped filename)
        csv=$(ls -t "$outdir"/standardllm_results_${ds}_*.csv 2>/dev/null | head -1)
        if [ -n "$csv" ]; then
            if [ "$header_written" -eq 0 ]; then
                cp "$csv" "$merged"
                header_written=1
            else
                # Append data rows (skip header)
                tail -n +2 "$csv" >> "$merged"
            fi
            echo "  Merged: $csv ($(tail -n +2 "$csv" | wc -l | tr -d ' ') rows)"
        else
            echo "  WARN: No results CSV for ${ds}/${sn}"
        fi
    done

    if [ -f "$merged" ]; then
        row_ct=$(tail -n +2 "$merged" | wc -l | tr -d ' ')
        echo "  -> $merged ($row_ct total rows)"

        echo ""
        echo ">>> Aggregating SC majority vote for $(upper "$ds")..."
        python src/sc_majority_vote.py "$merged" 2>&1 | tee "$LOGDIR/agg_${ds}.log"
    else
        echo "  WARN: No merged CSV for $ds, skipping aggregation"
    fi
done

# ---- Wait for W1 jobs ----
echo ""
echo ">>> Waiting for W1 (debate) jobs..."
for pid in "${W1_PIDS[@]}"; do
    wait "$pid" || echo "WARN: W1 process $pid failed"
done
echo ">>> All W1 jobs finished."

# ---- Stop monitor and print final status ----
kill "$MONITOR_PID" 2>/dev/null
wait "$MONITOR_PID" 2>/dev/null
print_status

echo ""
echo "============================================================"
echo "W1 + W2 experiments complete!"
echo "============================================================"
echo ""
echo "Logs: $LOGDIR/"
for f in "$LOGDIR"/*.log; do
    fstatus="OK"
    if grep -qiE "Traceback|FATAL" "$f" 2>/dev/null; then
        fstatus="CHECK ERRORS"
    fi
    printf "  %-40s %s\n" "$(basename "$f")" "$fstatus"
done
echo ""
echo "SC merged results:"
ls -lh results/standardllm/sc_*_merged*.csv 2>/dev/null || echo "  (none found)"
echo ""
echo "Next steps:"
echo "  1. Aggregate debate transcripts:"
echo "     python src/step2_aggregate_transcripts_ver5_FREEZE.py"
echo "  2. Compare debate vs SC:"
echo "     python src/compare_debate_vs_sc.py \\"
echo "       --debate-csv <step2_aggregate.csv> \\"
echo "       --sc-per-case <sc_majority_vote_per_case.csv> \\"
echo "       --sc-raw <sc_merged.csv> \\"
echo "       --dataset nlsy97"
