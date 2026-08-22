"""
Forensic Scaling Audit: Inspects raw records from Phase 11 across 8K, 16K, 32K, and 64K scales.
Audits requested_vocab_size vs actual_vocab_size, model parameters, FLOPs, and vocabulary saturation.
"""

import json
from pathlib import Path

records_path = Path("benchmarks/phase_eleven_scaling_records.json")
if not records_path.exists():
    print(f"Error: {records_path} not found")
    exit(1)

with open(records_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data.get("all_records", [])

print("=" * 140)
print("PHASE ELEVEN: FORENSIC SCALING RECORD AUDIT")
print("=" * 140)
print(
    f"{'Scale (Req)':<12} | {'Seed':<6} | {'Model Architecture':<30} | {'True BPB':<10} | {'Token CE':<10} | {'B/Tok':<8} | {'Active %':<10} | {'>=6B %'}"
)
print("-" * 140)

for r in records:
    print(
        f"V={r['vocab_size']:<10} | {r['seed']:<6} | {r['model_name']:<30} | {r['true_lm_bpb']:<10.3f} | {r['token_ce_loss']:<10.3f} | {r['bytes_per_token']:<8.2f} | {r['active_vocab_pct']:<10.1f}% | {r['pct_ge_6b']:<6.1f}%"
    )
print("=" * 140)
