# Run Commands: EXPERIMENT_8 (Human-Anchor-Steered Mask2Former)

## 1. Local Smoke Test (macOS CPU / MPS)
To verify forward pass, loss calculation, backward pass, and evaluation:
```bash
python3 experiments/EXPERIMENT_8/scripts/train.py \
    --data_dir data/L3D \
    --anchor_json data/llm_annotate/train_biological_anchors_human.json \
    --out_dir experiments/EXPERIMENT_8/results_smoke \
    --epochs 1 \
    --batch_size 2 \
    --accum_steps 1 \
    --smoke_test \
    --device cpu
```

## 2. Server SLURM Execution (`gpu-a240`)
Submit the complete 60-epoch training and evaluation job to the cluster:
```bash
sbatch experiments/EXPERIMENT_8/scripts/run_server.sbatch
```

### Monitoring the Server Job:
```bash
# Check queue status
squeue -u khoalq

# Tail stdout log
tail -f /data/khoalq/logs/exp8_human_anchor_m2f_<JOB_ID>.out
```

## 3. Standalone Post-Hoc Evaluation & Diagnostics
To re-run evaluation or regenerate the 101 Patient 40 diagnostic montages from a trained checkpoint:
```bash
python3 experiments/EXPERIMENT_8/scripts/evaluate.py \
    --checkpoint /data/khoalq/checkpoints/exp8_human_anchor_m2f/best_model.pth \
    --data_dir /data/khoalq/data/L3D \
    --out_dir /data/khoalq/checkpoints/exp8_human_anchor_m2f \
    --device cuda
```
