"""Build a CSV comparing Jev's default-prompt vs. tailored-prompt accuracy
against the paper's reported numbers for the same (or closest apples-to-
apples) configuration.

Run from src/evaluation/ with PYTHONPATH set to the repo root, e.g.:

    cd src/evaluation
    PYTHONPATH=/mnt/d/Rowan/DiSC-AMC python3 build_jev_tailoring_csv.py
"""
import csv
import os

from src.evaluation.jev_eval import read_results, get_class_names
from src.evaluation.utils import sort_results_by_prompt, acc, clean_acc, get_unique_prompts

EXP_DIR = "/mnt/d/Rowan/DiSC-AMC/exp"

# Each entry fully describes one evaluation config (shared by the default
# and tailored runs — only model_name/output_dir differ between the two).
CONFIGS = [
    {
        "label": "SynthMod-10, Table VI baseline (centroid, stats)",
        "dataset_folder": "unlabeled_10k",
        "dataset_type": "own",
        "prediction_source": "centroid",
        "feature_type": "stats",
        "n_components": 0,
        "noise_mode": "noisySignal",
        "n_bins": 5,
        "top_k": 5,
        "prompt_version": "v1",
        "backbone": "dino",
        "default_dir": "unlabeled_10k_centroid_discret_prompts_jev_jev-1.13_v02",
        "tailored_dir": "unlabeled_10k_centroid_discret_prompts_jev_jev-1.13-tailored_v01",
        "paper_location": "Table VI (centroid candidate retrieval, RAG=false): "
                           "Gemini-2.5-Flash 45.5%, Gemini-2.5-Pro 51.0%",
        "paper_accuracy": "45.5% (Flash) / 51.0% (Pro)",
    },
    {
        "label": "SynthMod-10, best config (faiss_filled + embeddings)",
        "dataset_folder": "unlabeled_10k",
        "dataset_type": "own",
        "prediction_source": "faiss_filled",
        "feature_type": "embeddings",
        "n_components": 10,
        "noise_mode": "noisySignal",
        "n_bins": 5,
        "top_k": 5,
        "prompt_version": "v1",
        "backbone": "dino",
        "default_dir": "unlabeled_10k_faiss_filled_emb10_discret_prompts_jev_jev-1.13_v01",
        "tailored_dir": "unlabeled_10k_faiss_filled_emb10_discret_prompts_jev_jev-1.13-tailored_v01",
        "paper_location": "Sec. VI-A / Fig. 4 & 5 (star annotation, training-free, FAISS+embeddings)",
        "paper_accuracy": "71.5% (best training-free cell, DeepSeek-R1-7B)",
    },
    {
        "label": "SynthMod-10, apples-to-apples w/ 71.5% DeepSeek run (plain faiss)",
        "dataset_folder": "unlabeled_10k",
        "dataset_type": "own",
        "prediction_source": "faiss",
        "feature_type": "embeddings",
        "n_components": 10,
        "noise_mode": "noisySignal",
        "n_bins": 5,
        "top_k": 5,
        "prompt_version": "v1",
        "backbone": "dino",
        "default_dir": "unlabeled_10k_faiss_emb10_discret_prompts_jev_jev-1.13_v01",
        "tailored_dir": "unlabeled_10k_faiss_emb10_discret_prompts_jev_jev-1.13-tailored_v01",
        "paper_location": "exp/comparisons/unlabeled_10k_faiss_emb10_disc_unsloth_"
                           "DeepSeek-R1-Distill-Qwen-7B_v01 (same config.json: faiss, "
                           "embeddings, RAG=false); cited in Sec. VI-A / Fig. 4 & 5",
        "paper_accuracy": "71.5% (143/200, DeepSeek-R1-Distill-Qwen-7B)",
    },
    {
        "label": "RadioML snr_20db, best config (faiss_filled_rml + embeddings)",
        "dataset_folder": "snr_20db",
        "dataset_type": "radioml",
        "prediction_source": "faiss_filled_rml",
        "feature_type": "embeddings",
        "n_components": 10,
        "noise_mode": "snr_20db",
        "n_bins": 5,
        "top_k": 5,
        "prompt_version": "v2",
        "backbone": "dino",
        "default_dir": "snr_20db_faiss_filled_rml_emb10_discret_prompts_jev_jev-1.13_v01",
        "tailored_dir": "snr_20db_faiss_filled_rml_emb10_discret_prompts_jev_jev-1.13-tailored_v01",
        "paper_location": "Table III (DiSC-AMC column, DINOv2 fine-tuned encoder, +20 dB row)",
        "paper_accuracy": "76.25% (183/240, DeepSeek-R1-Distill-Qwen-7B)",
    },
]


def _compute(output_dir: str, model_name: str, cfg: dict):
    if not os.path.isdir(os.path.join(EXP_DIR, output_dir)):
        return None
    try:
        results = read_results(
            dataset_folder=cfg["dataset_folder"],
            prompt_type="discret_prompts",
            model_name=model_name,
            noise_mode=cfg["noise_mode"],
            n_bins=cfg["n_bins"],
            top_k=cfg["top_k"],
            prediction_source=cfg["prediction_source"],
            feature_type=cfg["feature_type"],
            n_components=cfg["n_components"],
            ood_train_folder="",
            use_rag=False,
            rag_k=0,
            output_dir=os.path.join(EXP_DIR, output_dir),
            prompt_version=cfg["prompt_version"],
            backbone=cfg["backbone"],
        )
    except FileNotFoundError:
        return None
    n_unique = len(get_unique_prompts(results))
    sorted_results = sort_results_by_prompt(results)
    correct, total, accuracy = acc(sorted_results)
    _, _, clean_accuracy = clean_acc(sorted_results, get_class_names(cfg["dataset_type"]))
    return {
        "n_unique_prompts": n_unique,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "clean_accuracy": clean_accuracy,
    }


def main():
    rows = []
    for cfg in CONFIGS:
        default_metrics = _compute(cfg["default_dir"], "jev-1.13", cfg)
        tailored_metrics = _compute(cfg["tailored_dir"], "jev-1.13-tailored", cfg)

        def fmt(m):
            if m is None:
                return "", "", ""
            return (
                f"{m['correct']}/{m['total']}",
                f"{m['accuracy']:.4f}",
                f"{m['clean_accuracy']:.4f}",
            )

        d_frac, d_acc, d_clean = fmt(default_metrics)
        t_frac, t_acc, t_clean = fmt(tailored_metrics)

        delta = ""
        if default_metrics is not None and tailored_metrics is not None:
            delta = f"{(tailored_metrics['accuracy'] - default_metrics['accuracy']) * 100:+.1f}pp"

        rows.append({
            "Config": cfg["label"],
            "Dataset": cfg["dataset_folder"],
            "Prediction Source": cfg["prediction_source"],
            "Feature Type": cfg["feature_type"] + (f"{cfg['n_components']}" if cfg["feature_type"] == "embeddings" else ""),
            "Paper Location": cfg["paper_location"],
            "Paper Reported Accuracy": cfg["paper_accuracy"],
            "Jev Default-Prompt Acc (n/N)": d_frac,
            "Jev Default-Prompt Acc": d_acc,
            "Jev Default-Prompt Clean-Acc": d_clean,
            "Jev Tailored-Prompt Acc (n/N)": t_frac,
            "Jev Tailored-Prompt Acc": t_acc,
            "Jev Tailored-Prompt Clean-Acc": t_clean,
            "Delta (Tailored - Default)": delta,
        })

    out_path = os.path.join(EXP_DIR, "comparisons", "jev_prompt_tailoring_comparison.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    for r in rows:
        print("-" * 100)
        for k, v in r.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
