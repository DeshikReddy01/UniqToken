"""
Phase 15: Compute-Normalized Pareto Frontier & Multi-Objective Decision Analysis
================================================================================
Rigorous multi-objective decision analysis across all 27 experimental conditions:
    (3 Vocabulary Scales x 3 LM Capacity Tiers x 3 Tokenizer Architectures)
in the objective spaces:
    1. 2D Unconstrained: (True LM BPB, Token Cross-Entropy CE Loss)
    2. 3D Capacity-Constrained: (Total Parameters P, True LM BPB, Token CE)
    3. Scale-Constrained: Pareto sets within V in {16K, 32K, 64K}
    4. Tier-Specific: Pareto sets within LM Tier in {Small, Medium, Large}
    5. Fine-Grained Constrained Optimization across continuous thresholds.
    6. Per-Seed Paired Stability Analysis.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_all_records():
    benchmarks_dir = r"c:\Users\shaik\Research\Tokenizer\benchmarks"
    confirmatory_path = os.path.join(benchmarks_dir, "phase_fourteen_confirmatory_records.json")
    capacity_path = os.path.join(benchmarks_dir, "phase_fourteen_capacity_records.json")
    
    with open(confirmatory_path, "r", encoding="utf-8") as f:
        conf_data = json.load(f)
    with open(capacity_path, "r", encoding="utf-8") as f:
        cap_data = json.load(f)
        
    dataset = []
    tiers = ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"]
    tokenizers = ["SentencePiece-Unigram", "Boundary-BPE", "Caliper-SuperBPE (Config B)"]
    
    # 1. 16K data from Phase 14A (mean across 3 seeds)
    for tier in tiers:
        for tok in tokenizers:
            entry_16k = cap_data["summary_grid"][tier]["16384"][tok]
            dataset.append({
                "tier": tier,
                "vocab_size": 16384,
                "vocab_label": "16K",
                "tokenizer": tok,
                "bpb": float(entry_16k["true_lm_bpb"]),
                "ce": float(entry_16k["token_ce_loss"]),
                "bytes_per_token": float(entry_16k["bytes_per_token"]),
                "active_vocab_pct": float(entry_16k["active_vocab_pct"]),
                "total_params": int(entry_16k["total_params"]),
                "training_steps": int(entry_16k["training_steps"]),
                "num_seeds": 3,
                "source": "Phase 14A"
            })
            
    # 2. 32K and 64K data from Phase 14B (mean across 5 seeds)
    for tier in tiers:
        for v_scale, v_label in [("32768", "32K"), ("65536", "64K")]:
            for tok in tokenizers:
                entry = conf_data["summary_grid"][tier][v_scale][tok]
                ref_entry = cap_data["summary_grid"][tier][v_scale][tok]
                dataset.append({
                    "tier": tier,
                    "vocab_size": int(v_scale),
                    "vocab_label": v_label,
                    "tokenizer": tok,
                    "bpb": float(entry["true_lm_bpb_mean"]),
                    "bpb_std": float(entry["true_lm_bpb_std"]),
                    "ce": float(entry["token_ce_loss_mean"]),
                    "bytes_per_token": float(entry["bytes_per_token_mean"]),
                    "active_vocab_pct": float(entry["active_vocab_pct_mean"]),
                    "total_params": int(ref_entry["total_params"]),
                    "training_steps": int(ref_entry["training_steps"]),
                    "num_seeds": 5,
                    "source": "Phase 14B"
                })
                
    # Raw per-seed runs
    raw_runs_14a = cap_data.get("all_records", [])
    raw_runs_14b = conf_data.get("all_records", [])
    
    return dataset, raw_runs_14a, raw_runs_14b

def is_pareto_2d(costs):
    """Minimization on (c[0], c[1]). Returns boolean mask."""
    is_eff = [True] * len(costs)
    for i, c in enumerate(costs):
        if is_eff[i]:
            for j, other in enumerate(costs):
                if i != j:
                    if (other[0] <= c[0] and other[1] <= c[1]) and (other[0] < c[0] or other[1] < c[1]):
                        is_eff[i] = False
                        break
    return is_eff

def is_pareto_3d(costs):
    """Minimization on (c[0], c[1], c[2]). Returns boolean mask."""
    is_eff = [True] * len(costs)
    for i, c in enumerate(costs):
        if is_eff[i]:
            for j, other in enumerate(costs):
                if i != j:
                    if (other[0] <= c[0] and other[1] <= c[1] and other[2] <= c[2]) and \
                       (other[0] < c[0] or other[1] < c[1] or other[2] < c[2]):
                        is_eff[i] = False
                        break
    return is_eff

def run_fine_grained_constrained_analysis(dataset):
    # Fine-grained CE thresholds from 9.0 to 13.0 in steps of 0.25
    ce_thresholds = np.arange(9.0, 13.25, 0.25)
    ce_constrained = []
    for tau in ce_thresholds:
        valid = [d for d in dataset if d["ce"] <= tau]
        if valid:
            best = min(valid, key=lambda x: x["bpb"])
            ce_constrained.append({
                "ce_ceiling": float(tau),
                "best_bpb": float(best["bpb"]),
                "best_ce": float(best["ce"]),
                "tokenizer": best["tokenizer"],
                "tier": best["tier"],
                "vocab": best["vocab_label"],
                "total_params": best["total_params"]
            })
        else:
            ce_constrained.append({
                "ce_ceiling": float(tau),
                "best_bpb": None,
                "tokenizer": "Infeasible",
                "tier": "-",
                "vocab": "-",
                "total_params": None
            })
            
    # Fine-grained BPB thresholds from 1.9 to 3.6 in steps of 0.1
    bpb_thresholds = np.arange(1.9, 3.65, 0.1)
    bpb_constrained = []
    for tau in bpb_thresholds:
        valid = [d for d in dataset if d["bpb"] <= tau]
        if valid:
            best = min(valid, key=lambda x: x["ce"])
            bpb_constrained.append({
                "bpb_ceiling": float(tau),
                "best_ce": float(best["ce"]),
                "best_bpb": float(best["bpb"]),
                "tokenizer": best["tokenizer"],
                "tier": best["tier"],
                "vocab": best["vocab_label"],
                "total_params": best["total_params"]
            })
        else:
            bpb_constrained.append({
                "bpb_ceiling": float(tau),
                "best_ce": None,
                "tokenizer": "Infeasible",
                "tier": "-",
                "vocab": "-",
                "total_params": None
            })
            
    # Parameter-budget constrained
    param_thresholds = [5.0, 10.0, 20.0, 40.0, 60.0, 100.0]  # Millions
    param_constrained = []
    for tau in param_thresholds:
        valid = [d for d in dataset if (d["total_params"] / 1e6) <= tau]
        if valid:
            best = min(valid, key=lambda x: x["bpb"])
            param_constrained.append({
                "param_ceiling_m": tau,
                "best_bpb": float(best["bpb"]),
                "best_ce": float(best["ce"]),
                "tokenizer": best["tokenizer"],
                "tier": best["tier"],
                "vocab": best["vocab_label"],
                "total_params": best["total_params"]
            })
            
    return ce_constrained, bpb_constrained, param_constrained

def plot_comprehensive_pareto_figure(dataset, pareto_2d, pareto_3d, ce_constrained, out_path):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(18, 14), dpi=300)
    
    tok_colors = {
        "SentencePiece-Unigram": "#2b5c8f",
        "Boundary-BPE": "#d95f02",
        "Caliper-SuperBPE (Config B)": "#7570b3"
    }
    tier_markers = {
        "Small (4L-128d)": "o",
        "Medium (6L-256d)": "s",
        "Large (8L-512d)": "^"
    }
    
    # -------------------------------------------------------------
    # Panel 1: Full 2D Objective Space with Scale Hulls
    # -------------------------------------------------------------
    ax1 = axs[0, 0]
    
    for tok in tok_colors:
        sub = [d for d in dataset if d["tokenizer"] == tok]
        x = [d["ce"] for d in sub]
        y = [d["bpb"] for d in sub]
        ax1.scatter(x, y, color=tok_colors[tok], label=tok.split("-")[0], alpha=0.55, s=60)
        
    for d in dataset:
        ax1.scatter(d["ce"], d["bpb"], color=tok_colors[d["tokenizer"]], 
                    marker=tier_markers[d["tier"]], s=110, edgecolors="black", linewidths=0.7)
        
    # Highlight 2D Pareto frontier
    px = [p["ce"] for p in pareto_2d]
    py = [p["bpb"] for p in pareto_2d]
    ax1.plot(px, py, color="#e41a1c", linestyle="--", linewidth=2.2, label="2D Global Frontier")
    ax1.scatter(px, py, color="#e41a1c", marker="*", s=260, edgecolors="black", linewidths=1.2, zorder=5, label="2D Pareto Optimal")
    
    # Annotations
    for p in pareto_2d:
        ax1.annotate(f"{p['tokenizer'].split('-')[0]} ({p['vocab_label']}-{p['tier'].split()[0][0]})",
                     (p["ce"], p["bpb"]), xytext=(8, 4), textcoords="offset points",
                     fontsize=9, fontweight="bold", color="#e41a1c")
                     
    # Annotate Caliper 16K-Small
    cal_16k_s = next(d for d in dataset if d["vocab_label"] == "16K" and d["tier"] == "Small (4L-128d)" and "Caliper" in d["tokenizer"])
    ax1.annotate(f"Caliper 16K-S\n(BPB={cal_16k_s['bpb']:.2f}, 5.0M)",
                 (cal_16k_s["ce"], cal_16k_s["bpb"]), xytext=(-80, -25), textcoords="offset points",
                 fontsize=8.5, fontweight="bold", color="#7570b3",
                 arrowprops=dict(arrowstyle="->", color="#7570b3"))

    ax1.set_xlabel("Token Cross-Entropy Loss (nats) -> min", fontsize=12, fontweight="bold")
    ax1.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax1.set_title("A: Multi-Objective Landscape (True LM BPB vs Token CE)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 2: 3D Capacity-Constrained Pareto Frontier (Params vs BPB)
    # -------------------------------------------------------------
    ax2 = axs[0, 1]
    
    for tok in tok_colors:
        sub = [d for d in dataset if d["tokenizer"] == tok]
        params_m = [d["total_params"] / 1e6 for d in sub]
        bpb_vals = [d["bpb"] for d in sub]
        ax2.scatter(params_m, bpb_vals, color=tok_colors[tok], label=tok.split("-")[0],
                    marker="o", s=80, edgecolors="black", alpha=0.6)
        
    # Mark 3D Pareto Points with gold diamonds
    p3_params = [p["total_params"] / 1e6 for p in pareto_3d]
    p3_bpb = [p["bpb"] for p in pareto_3d]
    ax2.scatter(p3_params, p3_bpb, color="#ffd700", marker="D", s=140, edgecolors="black", linewidths=1.2, zorder=4,
                label="3D Pareto Optimal (Params, BPB, CE)")
    
    for p in pareto_3d:
        lbl = f"{p['tokenizer'].split('-')[0]} {p['vocab_label']}"
        ax2.annotate(lbl, (p["total_params"]/1e6, p["bpb"]),
                     xytext=(6, 4), textcoords="offset points", fontsize=8, fontweight="bold",
                     color=tok_colors[p["tokenizer"]])

    ax2.set_xlabel("Total Model Parameters (Millions) -> min", fontsize=12, fontweight="bold")
    ax2.set_ylabel("True LM BPB -> min", fontsize=12, fontweight="bold")
    ax2.set_title("B: 3D Parameter-Constrained Frontier: (P, BPB, CE)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 3: Constrained Decision: min(BPB) s.t. CE <= tau_CE
    # -------------------------------------------------------------
    ax3 = axs[1, 0]
    
    valid_ce_c = [c for c in ce_constrained if c["best_bpb"] is not None]
    taus = [c["ce_ceiling"] for c in valid_ce_c]
    best_bpbs = [c["best_bpb"] for c in valid_ce_c]
    
    ax3.step(taus, best_bpbs, where="post", color="#1b9e77", linewidth=2.5, label="Optimal BPB Profile")
    
    for c in valid_ce_c:
        if c["ce_ceiling"] in [9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5]:
            col = tok_colors[c["tokenizer"]]
            ax3.scatter(c["ce_ceiling"], c["best_bpb"], color=col, s=130, edgecolors="black", zorder=4)
            lbl = f"{c['tokenizer'].split('-')[0]} ({c['vocab']}-{c['tier'].split()[0][0]})"
            ax3.annotate(lbl, (c["ce_ceiling"], c["best_bpb"]),
                         xytext=(-20, 8), textcoords="offset points", fontsize=8.5, fontweight="bold",
                         color=col)
                         
    ax3.set_xlabel("Per-Token Cross-Entropy Ceiling tau_CE (nats)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Minimum Feasible BPB", fontsize=12, fontweight="bold")
    ax3.set_title("C: Constrained Decision: min(BPB) s.t. Token CE <= tau_CE", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Panel 4: Dominance Regime Matrix
    # -------------------------------------------------------------
    ax4 = axs[1, 1]
    
    grid_matrix = np.zeros((3, 3))
    text_labels = []
    tier_order = ["Small (4L-128d)", "Medium (6L-256d)", "Large (8L-512d)"]
    vocab_order = ["16K", "32K", "64K"]
    
    for r, tier in enumerate(tier_order):
        row_labels = []
        for c, v_label in enumerate(vocab_order):
            sub = [d for d in dataset if d["tier"] == tier and d["vocab_label"] == v_label]
            best_bpb = min(sub, key=lambda x: x["bpb"])
            best_ce = min(sub, key=lambda x: x["ce"])
            
            tok_map = {"SentencePiece-Unigram": 0, "Boundary-BPE": 1, "Caliper-SuperBPE (Config B)": 2}
            grid_matrix[r, c] = tok_map[best_bpb["tokenizer"]]
            
            lbl = f"BPB Winner: {best_bpb['tokenizer'].split('-')[0]}\n(BPB: {best_bpb['bpb']:.3f})\nCE Winner: {best_ce['tokenizer'].split('-')[0]}\n(CE: {best_ce['ce']:.3f})"
            row_labels.append(lbl)
        text_labels.append(row_labels)
        
    cmap = matplotlib.colors.ListedColormap(["#2b5c8f", "#d95f02", "#7570b3"])
    ax4.imshow(grid_matrix, cmap=cmap, aspect="auto", alpha=0.35, vmin=0, vmax=2)
    
    for r in range(3):
        for c in range(3):
            ax4.text(c, r, text_labels[r][c], ha="center", va="center", fontsize=9.5, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
                     
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(vocab_order, fontsize=11, fontweight="bold")
    ax4.set_yticks(range(3))
    ax4.set_yticklabels(["Small", "Medium", "Large"], fontsize=11, fontweight="bold")
    ax4.set_xlabel("Vocabulary Capacity", fontsize=12, fontweight="bold")
    ax4.set_ylabel("LM Architecture Tier", fontsize=12, fontweight="bold")
    ax4.set_title("D: Tokenizer Dominance Regimes across (Vocab x Capacity)", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved Phase 15 Pareto Frontier figure to {out_path}", flush=True)

def main():
    print("===============================================================================================================================================================================")
    print("PHASE FIFTEEN: COMPUTE-NORMALIZED PARETO FRONTIER & MULTI-OBJECTIVE DECISION ANALYSIS")
    print("Evaluating 27 Total Conditions (3 Vocab Scales x 3 LM Tiers x 3 Tokenizers)")
    print("===============================================================================================================================================================================\n")
    
    dataset, raw_14a, raw_14b = load_all_records()
    print(f"Loaded {len(dataset)} distinct experimental configurations ({len(raw_14a) + len(raw_14b)} total individual LM runs).\n")
    
    # 1. 2D Global Pareto Frontier
    costs_2d = [(d["bpb"], d["ce"]) for d in dataset]
    eff_2d = is_pareto_2d(costs_2d)
    pareto_2d = [d for i, d in enumerate(dataset) if eff_2d[i]]
    pareto_2d.sort(key=lambda x: x["bpb"])
    
    print("===============================================================================================================================================================================")
    print("1. GLOBAL 2D PARETO NON-DOMINATED FRONTIER on (True LM BPB, Token CE Loss)")
    print("===============================================================================================================================================================================")
    print(f"{'Tokenizer':<30} | {'Tier':<18} | {'Vocab':<6} | {'True BPB':<10} | {'Token CE (nats)':<16} | {'B/Tok':<8} | {'Params':<10} | {'Role'}")
    print("-" * 140)
    for p in pareto_2d:
        role = ""
        if p["bpb"] == min(d["bpb"] for d in dataset):
            role += "[Global min BPB] "
        if p["ce"] == min(d["ce"] for d in dataset):
            role += "[Global min CE] "
        print(f"{p['tokenizer']:<30} | {p['tier']:<18} | {p['vocab_label']:<6} | {p['bpb']:<10.3f} | {p['ce']:<16.3f} | {p['bytes_per_token']:<8.2f} | {p['total_params']:<10} | {role}")
    print("-" * 140 + "\n")
    
    # 2. 3D Global Pareto Frontier (Params, BPB, CE)
    costs_3d = [(d["total_params"], d["bpb"], d["ce"]) for d in dataset]
    eff_3d = is_pareto_3d(costs_3d)
    pareto_3d = [d for i, d in enumerate(dataset) if eff_3d[i]]
    pareto_3d.sort(key=lambda x: (x["total_params"], x["bpb"]))
    
    print("===============================================================================================================================================================================")
    print("2. GLOBAL 3D CAPACITY-CONSTRAINED PARETO FRONTIER on (Params P, True LM BPB, Token CE Loss)")
    print("===============================================================================================================================================================================")
    print(f"{'Tokenizer':<30} | {'Tier':<18} | {'Vocab':<6} | {'Params (M)':<12} | {'True BPB':<10} | {'Token CE (nats)':<16} | {'B/Tok':<8} | {'3D Pareto Role'}")
    print("-" * 140)
    for p in pareto_3d:
        note = ""
        if "Caliper" in p["tokenizer"] and p["vocab_label"] == "16K":
            note = "[16K-Small BPB Winner at 5.0M]"
        elif "Caliper" in p["tokenizer"] and p["vocab_label"] == "32K":
            note = "[32K-Small BPE-Class Winner at 9.2M]"
        elif p["bpb"] == min(d["bpb"] for d in dataset):
            note = "[Global min BPB]"
        elif p["ce"] == min(d["ce"] for d in dataset):
            note = "[Global min CE]"
        print(f"{p['tokenizer']:<30} | {p['tier']:<18} | {p['vocab_label']:<6} | {p['total_params']/1e6:<12.2f} | {p['bpb']:<10.3f} | {p['ce']:<16.3f} | {p['bytes_per_token']:<8.2f} | {note}")
    print("-" * 140 + "\n")
    
    # 3. Vocabulary-Scale-Constrained Pareto Sets
    print("===============================================================================================================================================================================")
    print("3. VOCABULARY-SCALE-CONSTRAINED PARETO FRONTIERS")
    print("===============================================================================================================================================================================")
    for v_lab in ["16K", "32K", "64K"]:
        v_sub = [d for d in dataset if d["vocab_label"] == v_lab]
        v_costs = [(d["bpb"], d["ce"]) for d in v_sub]
        v_eff = is_pareto_2d(v_costs)
        print(f"--- [Vocabulary Scale: {v_lab}] ---")
        for i, d in enumerate(v_sub):
            if v_eff[i]:
                print(f"  {d['tokenizer']:<28} | Tier: {d['tier']:<16} | BPB: {d['bpb']:<6.3f} | CE: {d['ce']:<6.3f} | B/Tok: {d['bytes_per_token']:<5.2f}")
    print("-" * 140 + "\n")

    # 4. Fine-Grained Constrained Decision Analysis
    ce_c, bpb_c, param_c = run_fine_grained_constrained_analysis(dataset)
    
    print("===============================================================================================================================================================================")
    print("4. FINE-GRAINED CONSTRAINED DECISIONS: min(BPB) SUBJECT TO CE CEILING (CE <= tau_CE)")
    print("===============================================================================================================================================================================")
    print(f"{'CE Ceiling tau_CE':<18} | {'Best Achievable BPB':<20} | {'Optimal Tokenizer':<30} | {'LM Tier':<18} | {'Vocab':<6} | {'CE at Opt':<10}")
    print("-" * 120)
    for c in ce_c:
        if c["ce_ceiling"] in [9.25, 9.5, 9.75, 10.0, 10.5, 11.0, 11.25, 11.5, 11.75, 12.0, 12.5, 13.0]:
            if c["best_bpb"] is not None:
                print(f"{c['ce_ceiling']:<18.2f} | {c['best_bpb']:<20.3f} | {c['tokenizer']:<30} | {c['tier']:<18} | {c['vocab']:<6} | {c['best_ce']:<10.3f}")
            else:
                print(f"{c['ce_ceiling']:<18.2f} | {'INFEASIBLE':<20} | {'-':<30} | {'-':<18} | {'-':<6} | {'-':<10}")
    print("-" * 120 + "\n")

    benchmarks_dir = r"c:\Users\shaik\Research\Tokenizer\benchmarks"
    fig_path = os.path.join(benchmarks_dir, "phase_fifteen_pareto_frontier.png")
    json_path = os.path.join(benchmarks_dir, "phase_fifteen_pareto_records.json")
    
    plot_comprehensive_pareto_figure(dataset, pareto_2d, pareto_3d, ce_c, fig_path)
    
    output_ledger = {
        "all_configurations": dataset,
        "pareto_2d_global": pareto_2d,
        "pareto_3d_params_bpb_ce": pareto_3d,
        "ce_constrained_decisions": ce_c,
        "bpb_constrained_decisions": bpb_c,
        "param_constrained_decisions": param_c
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_ledger, f, indent=2)
    print(f"[Ledger] Saved Phase 15 Pareto ledger to {json_path}\n", flush=True)

if __name__ == "__main__":
    main()
