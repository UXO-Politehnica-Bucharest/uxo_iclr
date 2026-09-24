#!/bin/bash
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")"
if [ -n "${CONDA_SH:-}" ]; then
    source "$CONDA_SH"
    conda activate "${CONDA_ENV:-lorentztim}"
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    export HF_TOKEN=$(grep '^HF_TOKEN=' .env | cut -d'=' -f2-)
fi

mkdir -p logs_final results_run
FAILED_STAGES=()
STAGE_NUM=0

run_stage() {
    local name="$1"
    shift
    STAGE_NUM=$((STAGE_NUM + 1))
    local log="logs_final/${STAGE_NUM}_${name}.log"
    echo ""
    echo "STAGE ${STAGE_NUM}: ${name}"
    echo "  cmd: $*"
    echo "  log: ${log}"
    echo "  started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if "$@" > "$log" 2>&1; then
        echo "STAGE ${STAGE_NUM} (${name}): OK  ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    else
        echo "STAGE ${STAGE_NUM} (${name}): FAILED (exit $?) -- see ${log}"
        FAILED_STAGES+=("${STAGE_NUM}_${name}")
    fi
}

run_stage "search_psi_dinov3" \
    python benchmark/search_and_update_psi.py --device cuda --backbone dinov3 --num-episodes 1000 \
    --search-trials 25 --val-episodes 30

run_stage "table2_dinov3" \
    python benchmark/table2_full_scale.py --device cuda --backbone dinov3 \
    --max-points 3000 --output-dir results_run/table2_dinov3

run_stage "table3_main_dinov3" \
    python benchmark/export_full_results.py --device cuda --backbone dinov3 \
    --num-episodes 1000 --shots 1,3,5,10 --output-dir results_run/table3_dinov3

run_stage "extended_shots_dinov3" \
    python benchmark/export_full_results.py --device cuda --backbone dinov3 \
    --num-episodes 1000 --shots 15,20,25,30,35,40 --output-dir results_run/extended_shots_dinov3

run_stage "table5_ablations_dinov3" \
    python benchmark/run_ablations_full.py --device cuda --backbone dinov3 \
    --num-episodes 1000 --shots 1,3,5,10 --output-dir results_run/table5_dinov3

run_stage "class_group_dinov3" \
    python benchmark/class_group_analysis.py --device cuda --backbone dinov3 \
    --shots 1,3,5,10 --num-episodes 1000 --output-dir results_run/class_group_dinov3

run_stage "imbalance_dinov3" \
    python benchmark/run_imbalance_full.py --device cuda --backbone dinov3 \
    --num-episodes 1000 --shot 5 --alphas 1.0,0.5,0.1 --output-dir results_run/imbalance_dinov3

run_stage "table3_clip_appendix" \
    python benchmark/export_full_results.py --device cuda --backbone clip \
    --num-episodes 1000 --shots 1,3,5,10 --force-search \
    --search-trials 25 --val-episodes 30 --output-dir results_run/table3_clip

for ds in cub200 fgvc_aircraft; do
    for bb in dinov3 clip; do
        run_stage "crossdomain_${ds}_${bb}" \
            python benchmark/cross_domain_eval.py --device cuda --dataset "$ds" --backbone "$bb" \
            --num-episodes 1000 --shots 1,3,5,10 --search-trials 25 --val-episodes 30 \
            --output-dir "results_run/crossdomain_${ds}_${bb}"
    done
done

for bb in dinov3 clip; do
    run_stage "crossdomain_tiered_imagenet_${bb}" \
        python benchmark/cross_domain_eval.py --device cuda --dataset tiered_imagenet --backbone "$bb" \
        --num-episodes 1000 --shots 1,3,5,10 --search-trials 25 --val-episodes 30 \
        --base-mean-max-per-class 50 \
        --output-dir "results_run/crossdomain_tiered_imagenet_${bb}"
done

run_stage "significance_table3_dinov3" \
    python benchmark/significance.py --scores-csv results_run/table3_dinov3/per_episode_scores.csv \
    --shots 1,3,5,10 --reference "TIM (Euclidean)" --output-dir results_run/table3_dinov3

run_stage "significance_extended_shots_dinov3" \
    python benchmark/significance.py --scores-csv results_run/extended_shots_dinov3/per_episode_scores.csv \
    --shots 15,20,25,30,35,40 --reference "TIM (Euclidean)" --candidates "LorentzTIM" \
    --output-dir results_run/extended_shots_dinov3

run_stage "significance_table3_clip" \
    python benchmark/significance.py --scores-csv results_run/table3_clip/per_episode_scores.csv \
    --shots 1,3,5,10 --reference "TIM (Euclidean)" --output-dir results_run/table3_clip

for ds in cub200 fgvc_aircraft tiered_imagenet; do
    for bb in dinov3 clip; do
        run_stage "significance_crossdomain_${ds}_${bb}" \
            python benchmark/significance.py \
            --scores-csv "results_run/crossdomain_${ds}_${bb}/per_episode_scores.csv" \
            --shots 1,3,5,10 --reference "TIM (Euclidean)" --candidates "LorentzTIM" \
            --output-dir "results_run/crossdomain_${ds}_${bb}"
    done
done

echo ""
echo "ALL STAGES DONE: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ${#FAILED_STAGES[@]} -eq 0 ]; then
    echo "No failed stages."
else
    echo "FAILED STAGES (${#FAILED_STAGES[@]}): ${FAILED_STAGES[*]}"
    echo "Check the corresponding logs_final/ files for each."
fi

