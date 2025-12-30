from __future__ import annotations

"""Run chain-of-thought free-form QA evaluation for RWKV models."""

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import shutil
from typing import Sequence

from src.eval.metrics.free_response import (
    compute_pass_at_k,
    compute_avg_at_k,
    evaluate_free_response,
    FreeResponseEvaluation,
)
from src.eval.results.layout import (
    eval_details_path,
    jsonl_path,
    param_search_records_path,
    param_search_trial_path,
    write_scores_json,
)
from src.eval.scheduler.dataset_resolver import resolve_or_prepare_dataset
from src.eval.scheduler.dataset_utils import infer_dataset_slug_from_path, canonical_slug
from src.eval.evaluators.free_response import (
    FreeResponsePipeline,
    FreeResponsePipelineResult,
    DEFAULT_COT_SAMPLING,
    DEFAULT_FINAL_SAMPLING,
)
from src.infer.model import ModelLoadConfig
from src.infer.sampling import SamplingConfig


MAX_TRIALS_PER_MODE = 30
DEFAULT_PARAM_SEARCH_TRIALS = 30
# Free-response 默认只算 pass@1；高难数学集单独放宽
DEFAULT_PASS_K = (1,)
DEFAULT_AVG_K: tuple[int, ...] = ()
AIME_AVG_K = (16,)
MATH_500_AVG_K = (4,)
AIME_REPORT_PASS_K = (8,)
AIME_PASS_K = (1, 2, 4, 8, 16, 32, 64, 128, 256)
HARD_MATH_PASS_K_SLUGS = {
    "aime24_test",
    "aime25_test",
    "beyond_aime_test",
    "hmmt_feb25_test",
    "brumo25_test",
    "college_math",
    "hle_al",
}
AVG_K_OVERRIDES = {
    "math_500": MATH_500_AVG_K,
    "math_500_test": MATH_500_AVG_K,
}
HARD_MATH_AVG_K_OVERRIDES = {
    "aime24_test": AIME_AVG_K,
    "aime25_test": AIME_AVG_K,
}

# Math benchmarks默认使用的 sampling 参数
MATH_DEFAULT_COT_SAMPLING = SamplingConfig(
    max_generate_tokens=4096,
    temperature=0.55,
    top_k=66,
    top_p=0.79,
    alpha_presence=0.14,
    alpha_frequency=0.01,
    alpha_decay=0.997,
    pad_zero=True,
)
SEARCH_PASS_K_OVERRIDES = {
    "aime24_test": (4,),
    "aime25_test": (4,),
    "beyond_aime_test": (4,),
    "hmmt_feb25_test": (4,),
    "brumo25_test": (4,),
}

PARAM_SEARCH_DATASETS: set[str] = set()

NORMAL_COT_SCAN_SPACE: dict[str, tuple[float, ...]] = {
    "temperature": (0.3, 0.4, 0.6, 0.8),
    "top_p": (0.3, 0.4, 0.5, 0.6),
    "alpha_presence": (0.5, 1.0, 1.5, 2.0),
    "alpha_frequency": (0.1, 0.3, 0.5),
    "alpha_decay": (0.99,),
}

SIMPLE_COT_SCAN_SPACE: dict[str, tuple[float, ...]] = {
    "temperature": (0.8, 0.6, 0.4, 0.3),
    "noise": (1.0, 2.0, 3.0),
}


def _round_float(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _resolve_output_path(dataset: str, model_path: str, user_path: str | None) -> Path:
    if user_path:
        return Path(user_path).expanduser()
    env_path = os.environ.get("RWKV_SKILLS_LOG_PATH")
    if env_path:
        return Path(env_path).expanduser()
    slug = infer_dataset_slug_from_path(dataset)
    return jsonl_path(slug, is_cot=True, model_name=Path(model_path).stem)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RWKV free-form CoT evaluator")
    parser.add_argument("--model-path", required=True, help="Path to RWKV weights (.pth)")
    parser.add_argument("--dataset", required=True, help="JSONL dataset path")
    parser.add_argument("--device", default="cuda", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples for quick runs")
    parser.add_argument("--cot-max-tokens", type=int, help="Clamp CoT generation length")
    parser.add_argument("--final-max-tokens", type=int, help="Clamp final answer generation length")
    parser.add_argument("--output", help="Output JSONL path (defaults to results/completions layout)")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Scheduler compatibility flag: run a single-sample probe and skip scoring",
    )
    parser.add_argument(
        "--no-param-search",
        action="store_true",
        help="Disable automatic sampling parameter search even for datasets that enable it by default.",
    )
    parser.add_argument(
        "--param-search",
        action="store_true",
        help="Force-enable sampling parameter search for this run.",
    )
    parser.add_argument(
        "--param-search-trials",
        type=int,
        help=(
            f"Max number of parameter-scan trials per sample_mode (default {DEFAULT_PARAM_SEARCH_TRIALS}, capped at {MAX_TRIALS_PER_MODE}). "
            "Set to a small number for quick smoke tests."
        ),
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        action="append",
        help="pass@k values to generate for and compute (default: 1; AIME 默认扩展到 256)",
    )
    parser.add_argument(
        "--avg-k",
        type=int,
        action="append",
        help="avg@k values to compute from generated samples (defaults depend on dataset; math500 用 4，AIME 用 16)",
    )
    return parser.parse_args(argv)


def _sampling_config_to_dict(config: SamplingConfig) -> dict[str, object]:
    data = asdict(config)
    normalized: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, tuple):
            normalized[key] = [
                _round_float(item) if isinstance(item, float) else item for item in value
            ]
        elif isinstance(value, float):
            normalized[key] = _round_float(value)
        else:
            normalized[key] = value
    return normalized


def _should_enable_param_search(args: argparse.Namespace, dataset_slug: str) -> bool:
    # 全局禁用参数扫描；仅显式开启时允许
    return bool(getattr(args, "param_search", False))


def _resolve_param_search_trials(args: argparse.Namespace) -> int:
    if args.param_search_trials and args.param_search_trials > 0:
        return min(int(args.param_search_trials), MAX_TRIALS_PER_MODE)
    env_val = os.environ.get("RWKV_PARAM_SEARCH_TRIALS")
    if env_val:
        try:
            parsed = int(env_val)
            if parsed > 0:
                return min(parsed, MAX_TRIALS_PER_MODE)
        except ValueError:
            pass
    return min(DEFAULT_PARAM_SEARCH_TRIALS, MAX_TRIALS_PER_MODE)


def _round_params_dict(data: dict[str, object]) -> dict[str, object]:
    rounded: dict[str, object] = {}
    for key, value in data.items():
        if isinstance(value, float):
            rounded[key] = _round_float(value)
        else:
            rounded[key] = value
    return rounded


def _max_k(values: Sequence[int] | None) -> int:
    return max(values) if values else 0


def _resolve_pass_k(slug: str, args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (final_pass_k, search_pass_k)."""

    if args.pass_k:
        resolved = tuple(args.pass_k)
        return resolved, resolved
    lower_slug = canonical_slug(str(slug))
    final_pass_k = AIME_PASS_K if lower_slug in HARD_MATH_PASS_K_SLUGS else DEFAULT_PASS_K
    search_pass_k = SEARCH_PASS_K_OVERRIDES.get(lower_slug, final_pass_k)
    return final_pass_k, search_pass_k


def _resolve_avg_k(slug: str, args: argparse.Namespace) -> tuple[int, ...]:
    if args.avg_k:
        return tuple(args.avg_k)
    lower_slug = canonical_slug(str(slug))
    if lower_slug in HARD_MATH_AVG_K_OVERRIDES:
        return HARD_MATH_AVG_K_OVERRIDES[lower_slug]
    return AVG_K_OVERRIDES.get(lower_slug, DEFAULT_AVG_K)


def _report_pass_k(slug: str, final_pass_k: tuple[int, ...]) -> tuple[int, ...]:
    lower_slug = canonical_slug(str(slug))
    if lower_slug in {"aime24_test", "aime25_test"}:
        return AIME_REPORT_PASS_K
    return final_pass_k


def _report_avg_k(slug: str, final_avg_k: tuple[int, ...]) -> tuple[int, ...]:
    lower_slug = canonical_slug(str(slug))
    if lower_slug in HARD_MATH_AVG_K_OVERRIDES:
        return HARD_MATH_AVG_K_OVERRIDES[lower_slug]
    if lower_slug in AVG_K_OVERRIDES:
        return AVG_K_OVERRIDES[lower_slug]
    return final_avg_k


def _filter_metrics_by_k(metric_map: dict[str, float] | None, ks: tuple[int, ...], prefix: str) -> dict[str, float]:
    if not metric_map or not ks:
        return {}
    allowed = {int(k) for k in ks if int(k) > 0}
    filtered: dict[str, float] = {}
    for key, value in metric_map.items():
        if not key.startswith(prefix):
            continue
        try:
            suffix = int(key.split("@", 1)[1])
        except (ValueError, IndexError):
            continue
        if suffix in allowed:
            filtered[key] = value
    return filtered


def _run_single_eval(
    pipeline: FreeResponsePipeline,
    dataset_path: str,
    output_path: Path,
    *,
    cot_sampling: SamplingConfig,
    final_sampling: SamplingConfig,
    batch_size: int,
    sample_limit: int | None,
    pad_to_batch: bool,
    pass_k: tuple[int, ...],
    samples_per_task: int,
    probe_only: bool = False,
) -> tuple[FreeResponsePipelineResult, FreeResponseEvaluation | None]:
    result = pipeline.run(
        dataset_path=dataset_path,
        output_path=str(output_path),
        cot_sampling=cot_sampling,
        final_sampling=final_sampling,
        batch_size=batch_size,
        sample_limit=sample_limit,
        pad_to_batch=pad_to_batch,
        pass_k=pass_k,
        samples_per_task=samples_per_task,
        probe_only=probe_only,
    )
    if probe_only:
        return result, None
    evaluation = evaluate_free_response(
        output_path,
        dataset_path=dataset_path,
        eval_output_path=None,
        judge=None,
    )
    return result, evaluation


def _run_param_search(
    pipeline: FreeResponsePipeline,
    dataset_path: str,
    base_output_path: Path,
    dataset_slug: str,
    model_name: str,
    *,
    cot_sampling: SamplingConfig,
    final_sampling: SamplingConfig,
    batch_size: int,
    sample_limit: int | None,
    trials: int,
    pass_k: tuple[int, ...],
    samples_per_task: int,
) -> tuple[
    FreeResponsePipelineResult,
    FreeResponseEvaluation,
    dict[str, object],
    SamplingConfig,
    SamplingConfig,
]:
    def _extract_best_params(cfg: SamplingConfig) -> dict[str, object]:
        mode = (cfg.sample_mode or "normal").strip().lower()
        if mode == "simple":
            return {
                "sample_mode": "simple",
                "temperature": _round_float(cfg.temperature),
                "noise": _round_float(float(cfg.noise)),
            }
        return {
            "sample_mode": "normal",
            "temperature": _round_float(cfg.temperature),
            "top_p": _round_float(cfg.top_p),
            "alpha_presence": _round_float(cfg.alpha_presence),
            "alpha_frequency": _round_float(cfg.alpha_frequency),
            "alpha_decay": _round_float(cfg.alpha_decay),
        }

    per_mode_limit = int(trials) if trials and trials > 0 else DEFAULT_PARAM_SEARCH_TRIALS
    per_mode_limit = min(per_mode_limit, MAX_TRIALS_PER_MODE)
    if per_mode_limit <= 0:
        raise ValueError("参数扫描 trials 必须大于 0")

    search_records: list[dict[str, object]] = []
    best_tuple: tuple[FreeResponsePipelineResult, FreeResponseEvaluation, Path, SamplingConfig] | None = None
    best_trial_index: int | None = None
    best_by_mode: dict[str, tuple[int, float, str, SamplingConfig]] = {}
    trials_by_mode: dict[str, int] = {}

    def _run_trial(trial_idx: int, cot_cfg: SamplingConfig) -> FreeResponseEvaluation:
        nonlocal best_tuple, best_trial_index
        candidate_path = param_search_trial_path(
            dataset_slug,
            is_cot=True,
            model_name=model_name,
            trial_index=trial_idx,
        )
        candidate_path.unlink(missing_ok=True)
        print(f"🔍 参数扫描 trial {trial_idx} ({cot_cfg.sample_mode}): {candidate_path.name}")
        result = pipeline.run(
            dataset_path=dataset_path,
            output_path=str(candidate_path),
            cot_sampling=cot_cfg,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            pass_k=pass_k,
            samples_per_task=samples_per_task,
        )
        evaluation = evaluate_free_response(
            candidate_path,
            dataset_path=dataset_path,
            eval_output_path=None,
            judge=None,
        )
        record = {
            "trial": trial_idx,
            "sample_mode": cot_cfg.sample_mode,
            "exact_accuracy": evaluation.exact_accuracy,
            "samples": evaluation.samples,
            "log_path": str(candidate_path),
            "cot_sampling": _sampling_config_to_dict(cot_cfg),
            "final_sampling": _sampling_config_to_dict(final_sampling),
            "params": _extract_best_params(cot_cfg),
        }
        search_records.append(record)

        mode = (cot_cfg.sample_mode or "normal").strip().lower()
        trials_by_mode[mode] = trials_by_mode.get(mode, 0) + 1
        mode_best = best_by_mode.get(mode)
        if mode_best is None or float(evaluation.exact_accuracy) > float(mode_best[1]):
            best_by_mode[mode] = (trial_idx, float(evaluation.exact_accuracy), str(candidate_path), cot_cfg)

        if best_tuple is None or float(evaluation.exact_accuracy) > float(best_tuple[1].exact_accuracy):
            best_tuple = (result, evaluation, candidate_path, cot_cfg)
            best_trial_index = trial_idx
        return evaluation

    trial_idx = 0

    # 1) normal (旧版常规 sampling)：先扫 temperature/top_p，再扫 alpha_presence/alpha_frequency；alpha_decay 固定
    normal_remaining = per_mode_limit
    alpha_decay = float(NORMAL_COT_SCAN_SPACE["alpha_decay"][0])
    baseline_presence = float(NORMAL_COT_SCAN_SPACE["alpha_presence"][0])
    baseline_frequency = float(NORMAL_COT_SCAN_SPACE["alpha_frequency"][0])
    best_temp_top_p: tuple[float, float] | None = None
    best_temp_top_p_acc: float | None = None

    for temperature in NORMAL_COT_SCAN_SPACE["temperature"]:
        for top_p in NORMAL_COT_SCAN_SPACE["top_p"]:
            if normal_remaining <= 0:
                break
            trial_idx += 1
            evaluation = _run_trial(
                trial_idx,
                replace(
                    cot_sampling,
                    sample_mode="normal",
                    noise=0.0,
                    temperature=float(temperature),
                    top_p=float(top_p),
                    alpha_presence=baseline_presence,
                    alpha_frequency=baseline_frequency,
                    alpha_decay=alpha_decay,
                ),
            )
            normal_remaining -= 1
            acc = float(evaluation.exact_accuracy)
            if best_temp_top_p_acc is None or acc > best_temp_top_p_acc:
                best_temp_top_p_acc = acc
                best_temp_top_p = (float(temperature), float(top_p))
        if normal_remaining <= 0:
            break

    if normal_remaining > 0 and best_temp_top_p is not None:
        best_temperature, best_top_p = best_temp_top_p
        for alpha_presence in NORMAL_COT_SCAN_SPACE["alpha_presence"]:
            for alpha_frequency in NORMAL_COT_SCAN_SPACE["alpha_frequency"]:
                if normal_remaining <= 0:
                    break
                trial_idx += 1
                _ = _run_trial(
                    trial_idx,
                    replace(
                        cot_sampling,
                        sample_mode="normal",
                        noise=0.0,
                        temperature=best_temperature,
                        top_p=best_top_p,
                        alpha_presence=float(alpha_presence),
                        alpha_frequency=float(alpha_frequency),
                        alpha_decay=alpha_decay,
                    ),
                )
                normal_remaining -= 1
            if normal_remaining <= 0:
                break

    # 2) simple (albatross temp+noise)：扫 temperature/noise
    simple_remaining = per_mode_limit
    for temperature in SIMPLE_COT_SCAN_SPACE["temperature"]:
        for noise in SIMPLE_COT_SCAN_SPACE["noise"]:
            if simple_remaining <= 0:
                break
            trial_idx += 1
            _ = _run_trial(
                trial_idx,
                replace(
                    cot_sampling,
                    sample_mode="simple",
                    temperature=float(temperature),
                    noise=float(noise),
                ),
            )
            simple_remaining -= 1
        if simple_remaining <= 0:
            break

    if best_tuple is None:
        raise RuntimeError("参数扫描未生成任何可用 trial 结果")
    best_result, best_eval, best_path, best_cot_cfg = best_tuple
    best_final_cfg = final_sampling
    if base_output_path.exists():
        base_output_path.unlink(missing_ok=True)
    if best_path != base_output_path:
        shutil.copy2(best_path, base_output_path)
    best_result.output_path = base_output_path

    records_path = param_search_records_path(dataset_slug, is_cot=True, model_name=model_name)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as fh:
        for row in search_records:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")

    best_mode_payload: dict[str, object] = {}
    for mode, (trial_idx, acc, log_path, cot_cfg) in best_by_mode.items():
        best_mode_payload[mode] = {
            "best_trial": trial_idx,
            "best_exact_accuracy": _round_float(acc),
            "best_log_path": log_path,
            "best_params": _extract_best_params(cot_cfg),
            "best_cot_sampling": _sampling_config_to_dict(cot_cfg),
        }

    summary = {
        "enabled": True,
        "search_type": "grid",
        "max_trials_per_mode": MAX_TRIALS_PER_MODE,
        "trial_limit_per_mode": per_mode_limit,
        "trials": len(search_records),
        "trials_by_mode": {
            "normal": int(trials_by_mode.get("normal", 0)),
            "simple": int(trials_by_mode.get("simple", 0)),
        },
        "best_trial": int(best_trial_index) if best_trial_index is not None else None,
        "best_exact_accuracy": _round_float(best_eval.exact_accuracy),
        "best_params": _extract_best_params(best_cot_cfg),
        "best_log_path": str(best_path),
        "best_cot_sampling": _sampling_config_to_dict(best_cot_cfg),
        "best_final_sampling": _sampling_config_to_dict(best_final_cfg),
        "best_by_mode": best_mode_payload,
        "scan_space": {
            "normal": NORMAL_COT_SCAN_SPACE,
            "simple": SIMPLE_COT_SCAN_SPACE,
        },
        "records_path": str(records_path),
    }
    return best_result, best_eval, summary, best_cot_cfg, best_final_cfg

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dataset_path = resolve_or_prepare_dataset(args.dataset)
    except Exception as exc:
        print(f"❌ 数据集准备失败: {exc}")
        return 1
    slug = infer_dataset_slug_from_path(str(dataset_path))
    out_path = _resolve_output_path(str(dataset_path), args.model_path, args.output)
    config = ModelLoadConfig(weights_path=args.model_path, device=args.device)
    pipeline = FreeResponsePipeline(config)
    pass_k_final, pass_k_search = _resolve_pass_k(slug, args)
    avg_k_final = _resolve_avg_k(slug, args)
    report_pass_k = _report_pass_k(slug, pass_k_final)
    report_avg_k = _report_avg_k(slug, avg_k_final)

    lower_slug = canonical_slug(slug)
    base_cot_sampling = MATH_DEFAULT_COT_SAMPLING if lower_slug in HARD_MATH_PASS_K_SLUGS else DEFAULT_COT_SAMPLING
    cot_sampling = base_cot_sampling.clamp(args.cot_max_tokens)
    final_sampling = DEFAULT_FINAL_SAMPLING.clamp(args.final_max_tokens)
    batch_size = max(1, args.batch_size)
    sample_limit: int | None = args.max_samples if not args.probe_only else batch_size
    samples_per_task_final = max(_max_k(pass_k_final), _max_k(avg_k_final), 1)
    samples_per_task_search = max(_max_k(pass_k_search), _max_k(avg_k_final), 1)

    evaluation: FreeResponseEvaluation | None = None
    param_summary: dict[str, object] | None = None
    output_path = out_path
    if _should_enable_param_search(args, slug) and not args.probe_only:
        param_trials = _resolve_param_search_trials(args)
        search_pass_k = pass_k_final if args.pass_k else pass_k_search
        result, evaluation, param_summary, best_cot, best_final = _run_param_search(
            pipeline,
            dataset_path=str(dataset_path),
            base_output_path=out_path,
            dataset_slug=slug,
            model_name=Path(args.model_path).stem,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            trials=param_trials,
            pass_k=search_pass_k,
            samples_per_task=samples_per_task_search,
        )
        effective_cot = best_cot
        effective_final = best_final
        result, evaluation = _run_single_eval(
            pipeline,
            dataset_path=str(dataset_path),
            output_path=out_path,
            cot_sampling=effective_cot,
            final_sampling=effective_final,
            batch_size=batch_size,
            sample_limit=sample_limit,
            pad_to_batch=False,
            pass_k=pass_k_final,
            samples_per_task=samples_per_task_final,
        )
    else:
        result, evaluation = _run_single_eval(
            pipeline,
            dataset_path=str(dataset_path),
            output_path=out_path,
            cot_sampling=cot_sampling,
            final_sampling=final_sampling,
            batch_size=batch_size,
            sample_limit=sample_limit,
            pad_to_batch=bool(args.probe_only),
            pass_k=pass_k_final,
            samples_per_task=samples_per_task_final,
            probe_only=args.probe_only,
        )
        if args.probe_only:
            return 0

    eval_path = eval_details_path(slug, is_cot=True, model_name=Path(args.model_path).stem)
    evaluation = evaluate_free_response(
        out_path,
        dataset_path=str(dataset_path),
        eval_output_path=eval_path,
        judge=None,
    )
    pass_metrics_all = compute_pass_at_k(evaluation.rows, pass_k_final)
    avg_metrics_all = compute_avg_at_k(evaluation.rows, avg_k_final)
    task_details: dict[str, object] = {"eval_details_path": str(eval_path)}
    if param_summary:
        task_details["param_search"] = param_summary
    metrics_payload = {
        "exact_accuracy": evaluation.exact_accuracy,
        "judge_accuracy": evaluation.judge_accuracy,
    }
    pass_payload = _filter_metrics_by_k(pass_metrics_all, report_pass_k, "pass@") or (pass_metrics_all or {})
    if pass_payload:
        metrics_payload.update(pass_payload)
    avg_payload = _filter_metrics_by_k(avg_metrics_all, report_avg_k, "avg@") or (avg_metrics_all or {})
    if avg_payload:
        metrics_payload.update(avg_payload)
    if pass_metrics_all and pass_payload != pass_metrics_all:
        task_details["pass_curve"] = pass_metrics_all
    if avg_metrics_all and avg_payload != avg_metrics_all:
        task_details["avg_curve"] = avg_metrics_all

    score_path = write_scores_json(
        slug,
        is_cot=True,
        model_name=Path(args.model_path).stem,
        metrics=metrics_payload,
        samples=evaluation.samples,
        problems=result.problem_count,
        log_path=output_path,
        task="free_response",
        task_details=task_details,
    )
    print(f"✅ CoT free-form done: {result.sample_count} samples -> {result.output_path}")
    if param_summary:
        best_acc = param_summary.get("best_exact_accuracy")
        best_trial = param_summary.get("best_trial")
        best_log = param_summary.get("best_log_path")
        records_path = param_summary.get("records_path")
        if isinstance(best_acc, float):
            print(f"🔍 参数扫描最佳: trial {best_trial} (exact={best_acc:.4f}) -> {best_log}")
        else:
            print(f"🔍 参数扫描最佳: trial {best_trial} -> {best_log}")
        if records_path:
            print(f"    📁 详尽记录: {records_path}")
    print(f"📄 eval details saved: {eval_path}")
    print(f"📊 scores saved: {score_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
