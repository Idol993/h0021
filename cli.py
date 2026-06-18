import os
import sys
import csv
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from tqdm import tqdm

from presets import PRESETS, list_presets, get_preset, hex_to_ass_color
from merge_utils import merge_bilingual_srt, check_alignment, TIMESTAMP_TOLERANCE_MS
from caption_engine import (
    SUPPORTED_VIDEO_EXTS,
    transcribe_video,
    translate_srt,
    burn_subtitles,
    generate_preview,
    probe_video_info,
    count_srt_lines,
    count_translation_fails,
    count_merge_conflicts,
    human_size,
    read_state_file,
    write_state_file,
    stage_completed_in_state,
    logger as engine_logger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("caption_pipeline")


def _file_ok(path: Optional[str]) -> bool:
    return isinstance(path, str) and os.path.isfile(path) and os.path.getsize(path) > 0


def _derive_paths(video_path: str, out_dir: Optional[str] = None) -> Dict[str, str]:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    stem = Path(video_path).stem
    base_dir = out_dir if out_dir else video_dir
    os.makedirs(base_dir, exist_ok=True)
    return {
        "video": video_path,
        "video_dir": video_dir,
        "stem": stem,
        "base_dir": base_dir,
        "src_srt": os.path.join(base_dir, f"{stem}.zh.srt"),
        "tgt_srt": os.path.join(base_dir, f"{stem}.en.srt"),
        "bilingual_srt": os.path.join(base_dir, f"{stem}.bilingual.srt"),
        "output_video": os.path.join(base_dir, f"{stem}_subtitled.mp4"),
        "preview": os.path.join(base_dir, f"{stem}_preview.png"),
        "merge_conflict": os.path.join(base_dir, f"{stem}.merge_conflicts.log"),
        "translate_fail": os.path.join(base_dir, f"{stem}.translation_fails.log"),
        "state": os.path.join(base_dir, f".{stem}.state.json"),
    }


def _collect_videos(input_path: str, recursive: bool = False) -> List[str]:
    videos: List[str] = []
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in SUPPORTED_VIDEO_EXTS:
            videos.append(os.path.abspath(input_path))
        return videos

    if not os.path.isdir(input_path):
        return videos

    if recursive:
        for root, _, files in os.walk(input_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTS:
                    videos.append(os.path.abspath(os.path.join(root, f)))
    else:
        for f in os.listdir(input_path):
            full = os.path.join(input_path, f)
            if os.path.isfile(full):
                ext = os.path.splitext(f)[1].lower()
                if ext in SUPPORTED_VIDEO_EXTS:
                    videos.append(os.path.abspath(full))

    videos.sort()
    return videos


def _build_burn_overrides(
    font_override: Optional[str] = None,
    font_name: Optional[str] = None,
    font_size: Optional[int] = None,
    primary_color_hex: Optional[str] = None,
    outline_color_hex: Optional[str] = None,
    outline_width: Optional[int] = None,
    margin_v: Optional[int] = None,
    bg_bar: bool = False,
    bg_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if font_override:
        overrides["font_path"] = font_override
    if font_name:
        overrides["font_name"] = font_name
    if font_size is not None:
        overrides["font_size"] = font_size
    if primary_color_hex:
        overrides["primary_color_hex"] = primary_color_hex
    if outline_color_hex:
        overrides["outline_color_hex"] = outline_color_hex
    if outline_width is not None:
        overrides["outline_width"] = outline_width
    if margin_v is not None:
        overrides["margin_v"] = margin_v
    if bg_bar:
        overrides["background_bar"] = True
        overrides["bg_alpha"] = bg_alpha if bg_alpha is not None else 0.75
    elif bg_alpha is not None:
        overrides["bg_alpha"] = bg_alpha
    return overrides


STAGE_ORDER = ["transcribe", "translate", "merge", "burn"]
STAGE_OUTPUT_KEY = {
    "transcribe": "src_srt",
    "translate": "tgt_srt",
    "merge": "bilingual_srt",
    "burn": "output_video",
}


def _ensure_state_structure(state: Dict[str, Any], paths: Dict[str, str], video_info: Dict[str, Any]) -> Dict[str, Any]:
    state.setdefault("video", paths["video"])
    state.setdefault("video_basename", os.path.basename(paths["video"]))
    state.setdefault("started_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    state.setdefault("stages", {})
    for stg in STAGE_ORDER:
        state["stages"].setdefault(stg, {"status": "pending"})
    state.setdefault("meta", {})
    if video_info and "duration_seconds" in video_info:
        state["meta"]["input_duration_seconds"] = round(video_info["duration_seconds"], 2)
        state["meta"]["input_file_size_bytes"] = video_info.get("file_size_bytes", 0)
        state["meta"]["input_resolution"] = f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"
        state["meta"]["video_codec"] = video_info.get("video_codec", "")
        state["meta"]["audio_codec"] = video_info.get("audio_codec", "")
    return state


def _process_single_video(args: tuple) -> Dict[str, Any]:
    (
        video_path,
        out_dir,
        whisper_model,
        device,
        compute_type,
        whisper_language,
        source_lang,
        target_lang,
        preset_name,
        custom_style_json,
        preset_overrides,
        skip_transcribe,
        skip_translate,
        skip_merge,
        skip_burn,
        force_merge,
        merge_layout,
        merge_top_first,
        crf,
        ffmpeg_preset,
        extra_margin_v,
        beam_size,
        vad_filter,
    ) = args

    paths = _derive_paths(video_path, out_dir)
    t_start = time.time()

    report: Dict[str, Any] = {
        "video": os.path.basename(video_path),
        "video_path": video_path,
        "stages": {},
        "stage_reasons": {},
        "errors": [],
        "skipped_stages": [],
        "user_skipped_stages": [],
        "status": "OK",
        "output_files": {},
        "metrics": {},
    }

    for flag, name in [
        (skip_transcribe, "transcribe"),
        (skip_translate, "translate"),
        (skip_merge, "merge"),
        (skip_burn, "burn"),
    ]:
        if flag:
            report["user_skipped_stages"].append(name)

    video_info = {}
    try:
        video_info = probe_video_info(video_path)
    except Exception:
        video_info = {"duration_seconds": 0.0, "file_size_bytes": 0}

    report["metrics"]["input_duration_seconds"] = round(video_info.get("duration_seconds", 0.0), 2)
    report["metrics"]["input_file_size_bytes"] = int(video_info.get("file_size_bytes", 0))
    report["metrics"]["input_resolution"] = f"{video_info.get('width', 0)}x{video_info.get('height', 0)}"
    report["metrics"]["input_size_human"] = human_size(int(video_info.get("file_size_bytes", 0)))

    state: Dict[str, Any] = read_state_file(paths["state"]) or {}
    state = _ensure_state_structure(state, paths, video_info)

    if video_info.get("probe_error"):
        report["errors"].append(f"video probe failed: {video_info['probe_error']}")

    def _mark_stage_done(stg: str, output_path: str, elapsed: float, extra: Optional[Dict] = None):
        state["stages"][stg] = {
            "status": "done",
            "output": output_path,
            "time_seconds": round(elapsed, 2),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if extra:
            state["stages"][stg].update(extra)
        write_state_file(paths["state"], state)

    def _mark_stage_user_skip(stg: str):
        state["stages"][stg] = {
            "status": "skipped_by_user",
            "reason": "user passed --skip-" + stg,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_state_file(paths["state"], state)

    def _mark_stage_ready_skip(stg: str, reason: str):
        state["stages"][stg] = {
            "status": "skipped_cached",
            "reason": reason,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_state_file(paths["state"], state)

    def _mark_stage_fail(stg: str, err: str):
        state["stages"][stg] = {
            "status": "failed",
            "error": err,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_state_file(paths["state"], state)

    try:
        for stg, skip_flag, out_key in [
            ("transcribe", skip_transcribe, "src_srt"),
            ("translate", skip_translate, "tgt_srt"),
            ("merge", skip_merge, "bilingual_srt"),
            ("burn", skip_burn, "output_video"),
        ]:
            out_path = paths[out_key]

            if skip_flag:
                _mark_stage_user_skip(stg)
                report["stages"][stg] = None
                report["skipped_stages"].append(stg)
                report["stage_reasons"][stg] = "user --skip-" + stg
                continue

            if stage_completed_in_state(state, stg):
                st = state["stages"][stg]
                report["stages"][stg] = st.get("time_seconds")
                report["skipped_stages"].append(stg)
                report["stage_reasons"][stg] = "resume: already done"
                if st.get("output"):
                    report["output_files"][out_key] = st["output"]
                continue

            if _file_ok(out_path):
                report["stages"][stg] = None
                report["skipped_stages"].append(stg)
                report["stage_reasons"][stg] = "cached: output exists"
                report["output_files"][out_key] = out_path
                _mark_stage_ready_skip(stg, "output file already exists before run")
                continue

            if stg == "transcribe":
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 1/4: Transcribe")
                t0 = time.time()
                try:
                    transcribe_video(
                        video_path=video_path,
                        output_srt_path=paths["src_srt"],
                        model_size=whisper_model,
                        device=device,
                        compute_type=compute_type,
                        language=whisper_language,
                        beam_size=beam_size,
                        vad_filter=vad_filter,
                    )
                    elapsed = time.time() - t0
                    src_lines = count_srt_lines(paths["src_srt"])
                    report["stages"]["transcribe"] = round(elapsed, 2)
                    report["output_files"]["src_srt"] = paths["src_srt"]
                    report["metrics"]["src_srt_lines"] = src_lines
                    _mark_stage_done("transcribe", paths["src_srt"], elapsed, {"src_srt_lines": src_lines})
                except Exception as e:
                    report["errors"].append(f"transcribe: {e}")
                    _mark_stage_fail("transcribe", str(e))

            elif stg == "translate":
                if not _file_ok(paths["src_srt"]):
                    err = "translate aborted: source SRT missing (transcribe not done)"
                    report["errors"].append(err)
                    _mark_stage_fail("translate", err)
                    continue
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 2/4: Translate")
                t0 = time.time()
                try:
                    translate_srt(
                        input_srt_path=paths["src_srt"],
                        output_srt_path=paths["tgt_srt"],
                        source_lang=source_lang,
                        target_lang=target_lang,
                        fail_log_path=paths["translate_fail"],
                    )
                    elapsed = time.time() - t0
                    tgt_lines = count_srt_lines(paths["tgt_srt"])
                    fails = count_translation_fails(paths["translate_fail"], paths["src_srt"])
                    report["stages"]["translate"] = round(elapsed, 2)
                    report["output_files"]["tgt_srt"] = paths["tgt_srt"]
                    report["metrics"]["tgt_srt_lines"] = tgt_lines
                    report["metrics"]["translate_fail_lines"] = fails
                    _mark_stage_done(
                        "translate",
                        paths["tgt_srt"],
                        elapsed,
                        {"tgt_srt_lines": tgt_lines, "translate_fails": fails},
                    )
                except Exception as e:
                    report["errors"].append(f"translate: {e}")
                    _mark_stage_fail("translate", str(e))

            elif stg == "merge":
                if not (_file_ok(paths["src_srt"]) and _file_ok(paths["tgt_srt"])):
                    err = "merge aborted: source or target SRT missing"
                    report["errors"].append(err)
                    _mark_stage_fail("merge", err)
                    continue
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 3/4: Merge Bilingual")
                t0 = time.time()
                try:
                    ok, conflicts = merge_bilingual_srt(
                        src_srt_path=paths["src_srt"],
                        tgt_srt_path=paths["tgt_srt"],
                        output_path=paths["bilingual_srt"],
                        layout=merge_layout,
                        top_first=merge_top_first,
                        force=force_merge,
                        conflict_log_path=paths["merge_conflict"],
                    )
                    elapsed = time.time() - t0
                    bi_lines = count_srt_lines(paths["bilingual_srt"])
                    conflict_n = count_merge_conflicts(paths["merge_conflict"]) or len(conflicts)
                    report["stages"]["merge"] = round(elapsed, 2)
                    report["output_files"]["bilingual_srt"] = paths["bilingual_srt"]
                    report["metrics"]["bilingual_srt_lines"] = bi_lines
                    report["metrics"]["merge_conflicts"] = conflict_n
                    extra = {"bilingual_srt_lines": bi_lines, "merge_conflicts": conflict_n}
                    if conflicts:
                        report["stage_reasons"]["merge"] = f"{conflict_n} conflict(s) logged"
                    if not ok:
                        err = "merge aborted due to conflicts"
                        report["errors"].append(err)
                        extra["merge_result"] = "aborted"
                        _mark_stage_fail("merge", err)
                        continue
                    _mark_stage_done("merge", paths["bilingual_srt"], elapsed, extra)
                except Exception as e:
                    report["errors"].append(f"merge: {e}")
                    _mark_stage_fail("merge", str(e))

            elif stg == "burn":
                use_bilingual = _file_ok(paths["bilingual_srt"])
                use_src = _file_ok(paths["src_srt"])
                use_tgt = _file_ok(paths["tgt_srt"])
                if not (use_bilingual or use_src or use_tgt):
                    err = "burn aborted: no SRT files available"
                    report["errors"].append(err)
                    _mark_stage_fail("burn", err)
                    continue
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 4/4: Burn Subtitles")
                t0 = time.time()
                try:
                    burn_subtitles(
                        input_video_path=video_path,
                        output_video_path=paths["output_video"],
                        src_srt_path=paths["src_srt"] if not use_bilingual and use_src else None,
                        tgt_srt_path=paths["tgt_srt"] if not use_bilingual and use_tgt else None,
                        bilingual_srt_path=paths["bilingual_srt"] if use_bilingual else None,
                        preset_name=preset_name,
                        custom_style_json=custom_style_json,
                        preset_overrides=preset_overrides,
                        crf=crf,
                        preset=ffmpeg_preset,
                        extra_margin_v=extra_margin_v,
                    )
                    elapsed = time.time() - t0
                    out_size = os.path.getsize(paths["output_video"]) if _file_ok(paths["output_video"]) else 0
                    report["stages"]["burn"] = round(elapsed, 2)
                    report["output_files"]["output_video"] = paths["output_video"]
                    report["metrics"]["output_file_size_bytes"] = out_size
                    report["metrics"]["output_size_human"] = human_size(out_size)
                    _mark_stage_done(
                        "burn",
                        paths["output_video"],
                        elapsed,
                        {"output_size_bytes": out_size},
                    )
                except Exception as e:
                    report["errors"].append(f"burn: {e}")
                    _mark_stage_fail("burn", str(e))

    except Exception as e:
        report["errors"].append(f"FATAL: {str(e)}")
        report["traceback"] = traceback.format_exc()

    if not report.get("metrics", {}).get("src_srt_lines") and _file_ok(paths["src_srt"]):
        report["metrics"]["src_srt_lines"] = count_srt_lines(paths["src_srt"])
    if not report.get("metrics", {}).get("tgt_srt_lines") and _file_ok(paths["tgt_srt"]):
        report["metrics"]["tgt_srt_lines"] = count_srt_lines(paths["tgt_srt"])
    if not report.get("metrics", {}).get("bilingual_srt_lines") and _file_ok(paths["bilingual_srt"]):
        report["metrics"]["bilingual_srt_lines"] = count_srt_lines(paths["bilingual_srt"])
    if not report.get("metrics", {}).get("translate_fail_lines") and _file_ok(paths["translate_fail"]):
        report["metrics"]["translate_fail_lines"] = count_translation_fails(
            paths["translate_fail"], paths["src_srt"]
        )
    if not report.get("metrics", {}).get("merge_conflicts") and _file_ok(paths["merge_conflict"]):
        report["metrics"]["merge_conflicts"] = count_merge_conflicts(paths["merge_conflict"])
    if not report.get("metrics", {}).get("output_file_size_bytes") and _file_ok(paths["output_video"]):
        out_size = os.path.getsize(paths["output_video"])
        report["metrics"]["output_file_size_bytes"] = out_size
        report["metrics"]["output_size_human"] = human_size(out_size)

    report["total_time"] = round(time.time() - t_start, 2)
    report["status"] = "OK" if not report["errors"] else "FAIL"

    state["status"] = report["status"]
    state["total_time_seconds"] = report["total_time"]
    write_state_file(paths["state"], state)

    return report


REPORT_CSV_FIELDS = [
    "video",
    "status",
    "input_duration_min",
    "input_size",
    "input_resolution",
    "total_time_seconds",
    "transcribe_time",
    "translate_time",
    "merge_time",
    "burn_time",
    "transcribe_skip_reason",
    "translate_skip_reason",
    "merge_skip_reason",
    "burn_skip_reason",
    "src_srt_lines",
    "tgt_srt_lines",
    "bilingual_srt_lines",
    "translate_fail_lines",
    "merge_conflicts",
    "output_size",
    "output_video",
    "errors",
]


def _format_report_csv_row(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metrics", {})
    stages = r.get("stages", {})
    reasons = r.get("stage_reasons", {})
    dur = m.get("input_duration_seconds", 0) or 0
    row = {
        "video": r.get("video", ""),
        "status": r.get("status", "FAIL"),
        "input_duration_min": round(dur / 60, 2) if dur else "",
        "input_size": m.get("input_size_human", ""),
        "input_resolution": m.get("input_resolution", ""),
        "total_time_seconds": r.get("total_time", 0),
        "transcribe_time": stages.get("transcribe", ""),
        "translate_time": stages.get("translate", ""),
        "merge_time": stages.get("merge", ""),
        "burn_time": stages.get("burn", ""),
        "transcribe_skip_reason": reasons.get("transcribe", ""),
        "translate_skip_reason": reasons.get("translate", ""),
        "merge_skip_reason": reasons.get("merge", ""),
        "burn_skip_reason": reasons.get("burn", ""),
        "src_srt_lines": m.get("src_srt_lines", ""),
        "tgt_srt_lines": m.get("tgt_srt_lines", ""),
        "bilingual_srt_lines": m.get("bilingual_srt_lines", ""),
        "translate_fail_lines": m.get("translate_fail_lines", ""),
        "merge_conflicts": m.get("merge_conflicts", ""),
        "output_size": m.get("output_size_human", ""),
        "output_video": r.get("output_files", {}).get("output_video", ""),
        "errors": "; ".join(r.get("errors", [])),
    }
    return row


def _write_report_json(results: List[Dict[str, Any]], report_path: str, elapsed_total: float,
                       summary_extra: Optional[Dict[str, Any]] = None) -> None:
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(elapsed_total, 2),
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r.get("status") == "OK"),
            "fail": sum(1 for r in results if r.get("status") == "FAIL"),
            "resumed_skip_count": sum(
                1 for r in results if any(
                    "resume" in r.get("stage_reasons", {}).get(s, "")
                    for s in STAGE_ORDER
                )
            ),
            "cached_skip_count": sum(
                1 for r in results if any(
                    "cached" in r.get("stage_reasons", {}).get(s, "")
                    for s in STAGE_ORDER
                )
            ),
        },
        "videos": results,
    }
    if summary_extra:
        report["summary"].update(summary_extra)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _write_report_csv(results: List[Dict[str, Any]], report_path: str, elapsed_total: float) -> None:
    with open(report_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(_format_report_csv_row(r))


@click.group(invoke_without_command=True)
@click.version_option(version="1.1.0", prog_name="caption-pipeline")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        engine_logger.setLevel(logging.DEBUG)


@cli.command("transcribe", help="Extract audio from video and generate SRT via faster-whisper")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_srt", type=click.Path(dir_okay=False), default=None, help="Output SRT path")
@click.option("--whisper-model", type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]), default="base", help="Whisper model size")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="Compute device")
@click.option("--compute-type", type=click.Choice(["auto", "int8", "int8_float32", "int8_float16", "float16", "float32"]), default="auto", help="Compute precision")
@click.option("--language", default=None, help="Source language code (auto-detect if omitted)")
@click.option("--beam-size", type=int, default=5, help="Beam search size")
@click.option("--no-vad-filter", is_flag=True, help="Disable VAD pre-filtering")
@click.option("--no-postprocess", is_flag=True, help="Skip text post-processing")
def cmd_transcribe(video_path, output_srt, whisper_model, device, compute_type, language, beam_size, no_vad_filter, no_postprocess):
    if output_srt is None:
        stem = Path(video_path).stem
        output_srt = os.path.join(os.path.dirname(os.path.abspath(video_path)), f"{stem}.zh.srt")
    transcribe_video(
        video_path=video_path,
        output_srt_path=output_srt,
        model_size=whisper_model,
        device=device,
        compute_type=compute_type,
        language=language,
        beam_size=beam_size,
        vad_filter=not no_vad_filter,
        postprocess=not no_postprocess,
    )
    click.echo(f"Done -> {output_srt}")


@cli.command("translate", help="Translate an SRT file using deep-translator (Google)")
@click.argument("input_srt", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_srt", type=click.Path(dir_okay=False), default=None, help="Output SRT path")
@click.option("--source-lang", default="zh-CN", help="Source language (default zh-CN)")
@click.option("--target-lang", default="en", help="Target language (default en)")
@click.option("--fail-log", default=None, help="Path to failure log file")
@click.option("--no-batch", is_flag=True, help="Disable batch mode, translate line-by-line")
def cmd_translate(input_srt, output_srt, source_lang, target_lang, fail_log, no_batch):
    if output_srt is None:
        stem = Path(input_srt).stem
        output_srt = os.path.join(os.path.dirname(os.path.abspath(input_srt)), f"{stem}.en.srt")
    n = translate_srt(
        input_srt_path=input_srt,
        output_srt_path=output_srt,
        source_lang=source_lang,
        target_lang=target_lang,
        fail_log_path=fail_log,
        batch_mode=not no_batch,
    )
    click.echo(f"Translated {n} lines -> {output_srt}")


@cli.command("burn", help="Burn SRT subtitles into video (hard subs) via ffmpeg")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_video", type=click.Path(dir_okay=False), default=None, help="Output video path")
@click.option("--src-srt", default=None, help="Source-language SRT path")
@click.option("--tgt-srt", default=None, help="Target-language SRT path")
@click.option("--bilingual-srt", default=None, help="Pre-merged bilingual SRT (takes precedence)")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default="white", help="Built-in style preset")
@click.option("--style-json", default=None, help="Custom style JSON file path")
@click.option("--font", "font_override", default=None, help="Override font file path")
@click.option("--font-name", default=None, help="Override font name")
@click.option("--font-size", type=int, default=None, help="Override font size")
@click.option("--color", "primary_color_hex", default=None, help="Text color #RRGGBB (e.g. #FF0000)")
@click.option("--outline-color", "outline_color_hex", default=None, help="Outline color #RRGGBB")
@click.option("--outline", "outline_width", type=int, default=None, help="Outline thickness in pixels")
@click.option("--margin", "margin_v", type=int, default=None, help="Bottom margin in pixels")
@click.option("--bg-bar", is_flag=True, help="Enable semi-transparent background bar")
@click.option("--bg-alpha", type=float, default=None, help="Background bar opacity 0.0-1.0")
@click.option("--crf", type=int, default=23, help="x264 CRF (default 23, lower=better)")
@click.option("--ffmpeg-preset", type=click.Choice(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]), default="ultrafast", help="Encoding preset")
def cmd_burn(video_path, output_video, src_srt, tgt_srt, bilingual_srt, preset, style_json,
             font_override, font_name, font_size, primary_color_hex, outline_color_hex,
             outline_width, margin_v, bg_bar, bg_alpha, crf, ffmpeg_preset):
    if output_video is None:
        stem = Path(video_path).stem
        output_video = os.path.join(os.path.dirname(os.path.abspath(video_path)), f"{stem}_subtitled.mp4")
    overrides = _build_burn_overrides(
        font_override=font_override,
        font_name=font_name,
        font_size=font_size,
        primary_color_hex=primary_color_hex,
        outline_color_hex=outline_color_hex,
        outline_width=outline_width,
        margin_v=margin_v,
        bg_bar=bg_bar,
        bg_alpha=bg_alpha,
    )
    burn_subtitles(
        input_video_path=video_path,
        output_video_path=output_video,
        src_srt_path=src_srt,
        tgt_srt_path=tgt_srt,
        bilingual_srt_path=bilingual_srt,
        preset_name=preset,
        custom_style_json=style_json,
        preset_overrides=overrides if overrides else None,
        crf=crf,
        preset=ffmpeg_preset,
    )
    click.echo(f"Burned -> {output_video}")


@cli.command("preview", help="Generate a preview screenshot with subtitle overlay at a given timestamp")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--src-srt", default=None, help="Source-language SRT path")
@click.option("--tgt-srt", default=None, help="Target-language SRT path")
@click.option("--bilingual-srt", default=None, help="Pre-merged bilingual SRT (takes precedence)")
@click.option("-t", "--timestamp", default="00:00:05", help="Capture timestamp (HH:MM:SS or seconds)")
@click.option("-o", "--output", "output_image", type=click.Path(dir_okay=False), default=None, help="Output image path")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default="white", help="Built-in style preset")
@click.option("--style-json", default=None, help="Custom style JSON file path")
@click.option("--font", "font_override", default=None, help="Override font file path")
@click.option("--font-name", default=None, help="Override font name")
@click.option("--font-size", type=int, default=None, help="Override font size")
@click.option("--color", "primary_color_hex", default=None, help="Text color #RRGGBB")
@click.option("--outline-color", "outline_color_hex", default=None, help="Outline color #RRGGBB")
@click.option("--outline", "outline_width", type=int, default=None, help="Outline thickness")
@click.option("--margin", "margin_v", type=int, default=None, help="Bottom margin (px)")
@click.option("--bg-bar", is_flag=True, help="Enable semi-transparent background bar")
@click.option("--bg-alpha", type=float, default=None, help="Background bar opacity 0.0-1.0")
def cmd_preview(video_path, src_srt, tgt_srt, bilingual_srt, timestamp, output_image,
                preset, style_json, font_override, font_name, font_size, primary_color_hex,
                outline_color_hex, outline_width, margin_v, bg_bar, bg_alpha):
    if output_image is None:
        stem = Path(video_path).stem
        output_image = os.path.join(os.path.dirname(os.path.abspath(video_path)), f"{stem}_preview.png")
    overrides = _build_burn_overrides(
        font_override=font_override,
        font_name=font_name,
        font_size=font_size,
        primary_color_hex=primary_color_hex,
        outline_color_hex=outline_color_hex,
        outline_width=outline_width,
        margin_v=margin_v,
        bg_bar=bg_bar,
        bg_alpha=bg_alpha,
    )
    generate_preview(
        video_path=video_path,
        output_image_path=output_image,
        timestamp=timestamp,
        src_srt_path=src_srt,
        tgt_srt_path=tgt_srt,
        bilingual_srt_path=bilingual_srt,
        preset_name=preset,
        custom_style_json=style_json,
        preset_overrides=overrides if overrides else None,
    )
    click.echo(f"Preview -> {output_image}")


@cli.command("merge", help="Merge source + target SRT into bilingual SRT")
@click.option("--src", "src_srt", type=click.Path(exists=True, dir_okay=False), required=True, help="Source SRT")
@click.option("--tgt", "tgt_srt", type=click.Path(exists=True, dir_okay=False), required=True, help="Target SRT")
@click.option("-o", "--output", "output_srt", type=click.Path(dir_okay=False), required=True, help="Output bilingual SRT")
@click.option("--layout", type=click.Choice(["stack", "alternate", "source_only", "target_only"]), default="stack", help="Layout mode")
@click.option("--top", type=click.Choice(["source", "target"]), default="source", help="Top language in stack")
@click.option("--force", is_flag=True, help="Force merge despite drift")
@click.option("--conflict-log", default=None, help="Write conflicts to log file")
def cmd_merge(src_srt, tgt_srt, output_srt, layout, top, force, conflict_log):
    ok, conflicts = merge_bilingual_srt(
        src_srt_path=src_srt,
        tgt_srt_path=tgt_srt,
        output_path=output_srt,
        layout=layout,
        top_first=top,
        force=force,
        conflict_log_path=conflict_log,
    )
    if conflicts:
        click.echo(f"Merge had {len(conflicts)} conflict(s)")
    if not ok:
        click.echo("Merge aborted due to conflicts (use --force)")
        sys.exit(1)
    click.echo(f"Merged -> {output_srt}")


@cli.command("check-align", help="Check time-axis alignment between two SRTs")
@click.option("--a", "a_srt", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--b", "b_srt", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--tolerance-ms", type=float, default=TIMESTAMP_TOLERANCE_MS, help="Drift tolerance ms")
def cmd_check_align(a_srt, b_srt, tolerance_ms):
    ok, conflicts = check_alignment(a_srt, b_srt, tolerance_ms=tolerance_ms)
    click.echo(f"Aligned: {'YES' if ok else 'NO'}  ({len(conflicts)} conflict(s) beyond {tolerance_ms}ms)")
    for c in conflicts[:20]:
        click.echo(f"  Line {c.index}: start={c.start_diff_ms:.0f}ms end={c.end_diff_ms:.0f}ms")
    if len(conflicts) > 20:
        click.echo(f"  ... and {len(conflicts) - 20} more")


INSPECT_COLUMNS = [
    ("has_video", "input_video"),
    ("has_src_srt", "zh_srt"),
    ("has_tgt_srt", "en_srt"),
    ("has_bilingual_srt", "bilingual_srt"),
    ("has_output", "final_video"),
    ("has_preview", "preview"),
    ("has_state", "state_file"),
    ("state_status", "last_status"),
    ("src_lines", "zh_lines"),
    ("tgt_lines", "en_lines"),
    ("bi_lines", "bilingual_lines"),
    ("translate_fails", "translate_fails"),
    ("merge_conflicts", "merge_conflicts"),
    ("input_duration_min", "duration_min"),
    ("input_size", "input_size"),
    ("output_size", "output_size"),
    ("missing", "missing_artifacts"),
]


def _inspect_one(video_path: str, out_dir: Optional[str]) -> Dict[str, Any]:
    paths = _derive_paths(video_path, out_dir)
    info: Dict[str, Any] = {"video": os.path.basename(video_path), "video_path": video_path}

    has = lambda k: _file_ok(paths.get(k, ""))
    info["has_video"] = has("video")
    info["has_src_srt"] = has("src_srt")
    info["has_tgt_srt"] = has("tgt_srt")
    info["has_bilingual_srt"] = has("bilingual_srt")
    info["has_output"] = has("output_video")
    info["has_preview"] = has("preview")
    info["has_state"] = has("state")

    st = read_state_file(paths["state"]) if has("state") else None
    info["state_status"] = (st or {}).get("status", "") if isinstance(st, dict) else ""

    info["src_lines"] = count_srt_lines(paths["src_srt"]) if has("src_srt") else 0
    info["tgt_lines"] = count_srt_lines(paths["tgt_srt"]) if has("tgt_srt") else 0
    info["bi_lines"] = count_srt_lines(paths["bilingual_srt"]) if has("bilingual_srt") else 0
    info["translate_fails"] = count_translation_fails(paths["translate_fail"], paths["src_srt"]) if has("translate_fail") else 0
    info["merge_conflicts"] = count_merge_conflicts(paths["merge_conflict"]) if has("merge_conflict") else 0

    try:
        vinfo = probe_video_info(video_path)
        dur = vinfo.get("duration_seconds", 0.0)
        info["input_duration_min"] = round(dur / 60, 2) if dur else 0
        info["input_size"] = human_size(int(vinfo.get("file_size_bytes", 0)))
    except Exception:
        info["input_duration_min"] = 0
        info["input_size"] = "N/A"
    info["output_size"] = human_size(os.path.getsize(paths["output_video"])) if has("output_video") else "N/A"

    missing = []
    for flag, label in [
        (not info["has_src_srt"], "zh_srt"),
        (not info["has_tgt_srt"], "en_srt"),
        (not info["has_bilingual_srt"], "bilingual_srt"),
        (not info["has_output"], "final_video"),
    ]:
        if flag:
            missing.append(label)
    info["missing"] = ", ".join(missing)
    info["complete"] = len(missing) == 0

    return info


@cli.command("inspect", help="Inspect existing pipeline artifacts for videos, optionally export CSV")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default=None, help="Output directory (default same as video)")
@click.option("--recursive", is_flag=True, help="Recurse subdirectories")
@click.option("--export", "export_path", default=None, help="Write inspection CSV to this path")
@click.option("--show-complete/--hide-complete", default=True, help="Show/hide fully processed videos")
@click.option("--only-missing", is_flag=True, help="Only show videos with missing artifacts")
def cmd_inspect(input_path, output_dir, recursive, export_path, show_complete, only_missing):
    videos = _collect_videos(input_path, recursive=recursive)
    if not videos:
        click.echo("No supported video files found.")
        sys.exit(0)

    rows: List[Dict[str, Any]] = []
    click.echo(f"Inspecting {len(videos)} video(s)...\n")

    for vp in tqdm(videos, desc="Scan", unit="video"):
        info = _inspect_one(vp, output_dir)
        if only_missing and info["complete"]:
            continue
        rows.append(info)

    col_keys = [k for k, _ in INSPECT_COLUMNS]
    col_labels = [l for _, l in INSPECT_COLUMNS]
    click.echo()
    click.echo(" | ".join(["VIDEO"] + col_labels))
    click.echo("-" * (len("VIDEO") + 5 + sum(len(l) + 3 for l in col_labels)))

    complete = 0
    for info in rows:
        vals = []
        for k in col_keys:
            if k in ("has_video", "has_src_srt", "has_tgt_srt", "has_bilingual_srt", "has_output", "has_preview", "has_state"):
                vals.append("Y" if info.get(k) else "-")
            else:
                v = info.get(k, "")
                vals.append(str(v) if v not in (None, "") else "-")
        status = "OK" if info.get("complete") else "MISSING"
        if info.get("complete"):
            complete += 1
        if not show_complete and info.get("complete"):
            continue
        click.echo(" | ".join([info["video"][:30]] + vals) + f" [{status}]")

    click.echo()
    click.echo("=" * 60)
    click.echo(f"Total scanned:  {len(videos)}")
    click.echo(f"Complete:       {complete}")
    click.echo(f"Incomplete:     {len(videos) - complete}")

    if export_path:
        fieldnames = ["video"] + col_keys
        export_dir = os.path.dirname(os.path.abspath(export_path))
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
        with open(export_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            full_rows = []
            for vp in videos:
                info = _inspect_one(vp, output_dir)
                info["video"] = info["video"]
                full_rows.append(info)
            for info in full_rows:
                writer.writerow(info)
        click.echo(f"\nCSV report saved -> {export_path}")


@cli.command("pipe", help="End-to-end pipeline: transcribe -> translate -> merge -> burn")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default=None, help="Output directory (default same as each video)")
@click.option("--recursive", is_flag=True, help="Recurse subdirectories")
@click.option("--whisper-model", type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]), default="small", help="Whisper model size")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="Compute device")
@click.option("--compute-type", type=click.Choice(["auto", "int8", "int8_float32", "int8_float16", "float16", "float32"]), default="auto", help="Compute precision")
@click.option("--whisper-language", default=None, help="Force Whisper language")
@click.option("--source-lang", default="zh-CN", help="Source language for translation")
@click.option("--target-lang", default="en", help="Target language for translation")
@click.option("--preset", "preset_name", type=click.Choice(list(PRESETS.keys())), default="white", help="Burn-in style preset")
@click.option("--style-json", default=None, help="Custom style JSON")
@click.option("--merge-layout", type=click.Choice(["stack", "alternate", "source_only", "target_only"]), default="stack", help="Bilingual layout")
@click.option("--merge-top", type=click.Choice(["source", "target"]), default="source", help="Top language in stack")
@click.option("--force-merge", is_flag=True, help="Force merge despite drift")
@click.option("--skip-transcribe", is_flag=True, help="Skip transcribe stage")
@click.option("--skip-translate", is_flag=True, help="Skip translate stage")
@click.option("--skip-merge", is_flag=True, help="Skip merge stage")
@click.option("--skip-burn", is_flag=True, help="Skip burn stage")
@click.option("--workers", type=int, default=1, help="Parallel workers")
@click.option("--dry-run", is_flag=True, help="List planned actions and exit")
@click.option("--crf", type=int, default=23, help="x264 CRF")
@click.option("--ffmpeg-preset", type=click.Choice(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]), default="ultrafast", help="Encoding preset")
@click.option("--extra-margin", type=int, default=0, help="Extra bottom margin px")
@click.option("--beam-size", type=int, default=5, help="Whisper beam size")
@click.option("--no-vad-filter", is_flag=True, help="Disable Whisper VAD filter")
@click.option("--color", "primary_color_hex", default=None, help="Text color #RRGGBB burn stage")
@click.option("--outline-color", "outline_color_hex", default=None, help="Outline color #RRGGBB burn stage")
@click.option("--outline", "outline_width", type=int, default=None, help="Outline thickness burn stage")
@click.option("--font-size", type=int, default=None, help="Font size burn stage")
@click.option("--margin", "margin_v", type=int, default=None, help="Bottom margin px burn stage")
@click.option("--bg-bar", is_flag=True, help="Enable background bar burn stage")
@click.option("--bg-alpha", type=float, default=None, help="Background bar opacity 0.0-1.0")
@click.option("--report", "report_path", default=None, help="Report path (.json/.csv)")
@click.option("--report-format", type=click.Choice(["json", "csv", "both"]), default=None, help="Report format")
@click.option("--no-auto-report", is_flag=True, help="Don't auto-generate default JSON report")
def cmd_pipe(input_path, output_dir, recursive, whisper_model, device, compute_type,
             whisper_language, source_lang, target_lang, preset_name, style_json,
             merge_layout, merge_top, force_merge, skip_transcribe, skip_translate,
             skip_merge, skip_burn, workers, dry_run, crf, ffmpeg_preset, extra_margin,
             beam_size, no_vad_filter, primary_color_hex, outline_color_hex, outline_width,
             font_size, margin_v, bg_bar, bg_alpha, report_path, report_format, no_auto_report):

    all_videos = _collect_videos(input_path, recursive=recursive)
    if not all_videos:
        click.echo("No supported video files found.")
        sys.exit(0)

    if skip_transcribe:
        logger.info("Skipping transcribe stage (--skip-transcribe)")
    if skip_translate:
        logger.info("Skipping translate stage (--skip-translate)")
    if skip_merge:
        logger.info("Skipping merge stage (--skip-merge)")
    if skip_burn:
        logger.info("Skipping burn stage (--skip-burn)")

    overrides = _build_burn_overrides(
        primary_color_hex=primary_color_hex,
        outline_color_hex=outline_color_hex,
        outline_width=outline_width,
        font_size=font_size,
        margin_v=margin_v,
        bg_bar=bg_bar,
        bg_alpha=bg_alpha,
    )

    pending: List[str] = []
    skip_upfront: Dict[str, str] = {}

    for vp in all_videos:
        paths = _derive_paths(vp, output_dir)
        state = read_state_file(paths["state"])

        stage_results = {}
        for stg, skip, out_key in [
            ("transcribe", skip_transcribe, "src_srt"),
            ("translate", skip_translate, "tgt_srt"),
            ("merge", skip_merge, "bilingual_srt"),
            ("burn", skip_burn, "output_video"),
        ]:
            if skip:
                stage_results[stg] = "user_skip"
            elif stage_completed_in_state(state, stg):
                stage_results[stg] = "resume_skip"
            elif _file_ok(paths[out_key]):
                stage_results[stg] = "file_skip"
            else:
                stage_results[stg] = "need_run"

        if all(v in ("user_skip", "resume_skip", "file_skip") for v in stage_results.values()):
            reason_parts = []
            for s, v in stage_results.items():
                if v == "user_skip":
                    reason_parts.append(f"{s}=user")
                elif v == "resume_skip":
                    reason_parts.append(f"{s}=resume")
                elif v == "file_skip":
                    reason_parts.append(f"{s}=cached")
            skip_upfront[vp] = ", ".join(reason_parts)
        else:
            pending.append(vp)

    click.echo(f"Found {len(all_videos)} video(s).")
    click.echo(f"  Upfront-skipped (all stages done or user-skipped): {len(skip_upfront)}")
    click.echo(f"  Will process: {len(pending)} video(s) with {workers} worker(s).\n")

    if dry_run:
        for vp in all_videos:
            paths = _derive_paths(vp, output_dir)
            state = read_state_file(paths["state"])
            stages = []
            for stg, skip, out_key in [
                ("transcribe", skip_transcribe, "src_srt"),
                ("translate", skip_translate, "tgt_srt"),
                ("merge", skip_merge, "bilingual_srt"),
                ("burn", skip_burn, "output_video"),
            ]:
                if skip:
                    stages.append(f"SKIP(user:{stg[:3]})")
                elif stage_completed_in_state(state, stg):
                    stages.append(f"SKIP(resume:{stg[:3]})")
                elif _file_ok(paths[out_key]):
                    stages.append(f"SKIP(file:{stg[:3]})")
                else:
                    stages.append(stg.upper())
            prefix = "(SKIP) " if vp in skip_upfront else "(RUN)  "
            click.echo(f"  {prefix}{os.path.basename(vp)}: {' -> '.join(stages)}")
        return

    task_args_list = []
    for vp in pending:
        task_args_list.append((
            vp, output_dir, whisper_model, device, compute_type, whisper_language,
            source_lang, target_lang, preset_name, style_json,
            overrides if overrides else None,
            skip_transcribe, skip_translate, skip_merge, skip_burn,
            force_merge, merge_layout, merge_top, crf, ffmpeg_preset,
            extra_margin, beam_size, not no_vad_filter,
        ))

    results: List[Dict[str, Any]] = []
    t_start_all = time.time()

    pbar = tqdm(total=len(task_args_list), desc="Pipeline", unit="video")

    def _on_complete(res: Dict[str, Any]):
        results.append(res)
        base = res.get("video", "")
        errors = res.get("errors", [])
        status = "OK" if not errors else "FAIL"
        total = res.get("total_time", 0)
        pbar.set_postfix_str(f"last:{base[:28]} {status} {total:.1f}s")
        pbar.update(1)
        if errors:
            logger.error(f"{base}: {'; '.join(errors[:3])}")

    if workers <= 1:
        for args in task_args_list:
            r = _process_single_video(args)
            _on_complete(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_single_video, a): a for a in task_args_list}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception as e:
                    a = futures[fut]
                    r = {
                        "video": os.path.basename(a[0]),
                        "video_path": a[0],
                        "errors": [f"worker crash: {e}"],
                        "stages": {},
                        "stage_reasons": {},
                        "skipped_stages": [],
                        "user_skipped_stages": [],
                        "status": "FAIL",
                        "output_files": {},
                        "metrics": {},
                        "total_time": 0,
                    }
                _on_complete(r)

    pbar.close()

    for vp, reason in skip_upfront.items():
        paths = _derive_paths(vp, output_dir)
        video_info = {}
        try:
            video_info = probe_video_info(vp)
        except Exception:
            video_info = {"duration_seconds": 0.0, "file_size_bytes": 0}
        report_entry: Dict[str, Any] = {
            "video": os.path.basename(vp),
            "video_path": vp,
            "stages": {},
            "stage_reasons": {},
            "errors": [],
            "skipped_stages": [],
            "user_skipped_stages": [],
            "status": "OK",
            "output_files": {},
            "metrics": {
                "input_duration_seconds": round(video_info.get("duration_seconds", 0.0), 2),
                "input_file_size_bytes": int(video_info.get("file_size_bytes", 0)),
                "input_size_human": human_size(int(video_info.get("file_size_bytes", 0))),
                "input_resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
            },
            "total_time": 0,
            "note": "upfront-skipped: all stages complete",
        }
        for stg, skip, out_key in [
            ("transcribe", skip_transcribe, "src_srt"),
            ("translate", skip_translate, "tgt_srt"),
            ("merge", skip_merge, "bilingual_srt"),
            ("burn", skip_burn, "output_video"),
        ]:
            report_entry["stages"][stg] = None
            if skip:
                report_entry["stage_reasons"][stg] = "user --skip-" + stg
                report_entry["user_skipped_stages"].append(stg)
                report_entry["skipped_stages"].append(stg)
            elif _file_ok(paths[out_key]):
                report_entry["stage_reasons"][stg] = "cached: output exists before run"
                report_entry["skipped_stages"].append(stg)
                report_entry["output_files"][out_key] = paths[out_key]
        if _file_ok(paths["src_srt"]):
            report_entry["metrics"]["src_srt_lines"] = count_srt_lines(paths["src_srt"])
        if _file_ok(paths["tgt_srt"]):
            report_entry["metrics"]["tgt_srt_lines"] = count_srt_lines(paths["tgt_srt"])
        if _file_ok(paths["bilingual_srt"]):
            report_entry["metrics"]["bilingual_srt_lines"] = count_srt_lines(paths["bilingual_srt"])
        if _file_ok(paths["output_video"]):
            out_size = os.path.getsize(paths["output_video"])
            report_entry["metrics"]["output_file_size_bytes"] = out_size
            report_entry["metrics"]["output_size_human"] = human_size(out_size)
            report_entry["output_files"]["output_video"] = paths["output_video"]
        if _file_ok(paths["translate_fail"]):
            report_entry["metrics"]["translate_fail_lines"] = count_translation_fails(
                paths["translate_fail"], paths["src_srt"]
            )
        if _file_ok(paths["merge_conflict"]):
            report_entry["metrics"]["merge_conflicts"] = count_merge_conflicts(paths["merge_conflict"])
        results.append(report_entry)

    elapsed_total = time.time() - t_start_all
    results.sort(key=lambda r: r.get("video", ""))

    n_ok = sum(1 for r in results if r.get("status") == "OK")
    n_fail = sum(1 for r in results if r.get("status") == "FAIL")

    click.echo()
    click.echo("=" * 60)
    click.echo(f"Pipeline done in {elapsed_total:.1f}s.")
    click.echo(f"  Total videos: {len(results)}")
    click.echo(f"  OK:           {n_ok}")
    click.echo(f"  FAIL:         {n_fail}")
    click.echo()

    for r in results[:80]:
        base = r.get("video", "")
        tag = "OK  " if r.get("status") == "OK" else "FAIL"
        stages = r.get("stages", {})
        stage_parts = []
        for s in STAGE_ORDER:
            if s in (r.get("skipped_stages") or []):
                reason = r.get("stage_reasons", {}).get(s, "skip")
                short = reason.split(":")[0][:6] if reason else "skip"
                stage_parts.append(f"{s[:3]}={short}")
            elif stages.get(s) is not None:
                stage_parts.append(f"{s[:3]}={stages[s]}s")
            else:
                stage_parts.append(f"{s[:3]}=-")
        total = r.get("total_time", 0)
        click.echo(f"  [{tag}] {base[:38]:<38s}  total={total:>6}s  {'  '.join(stage_parts)}")
        for err in (r.get("errors") or [])[:2]:
            click.echo(f"          ! {err}")
    if len(results) > 80:
        click.echo(f"  ... and {len(results) - 80} more (see report)")

    outputs_to_write: List[Tuple[str, str]] = []

    if report_path:
        fmts: List[str] = []
        if report_format:
            fmts = [report_format] if report_format != "both" else ["json", "csv"]
        else:
            ext = os.path.splitext(report_path)[1].lower()
            if ext == ".csv":
                fmts = ["csv"]
            else:
                fmts = ["json"]
        base_report = os.path.splitext(report_path)[0]
        report_dir = os.path.dirname(os.path.abspath(report_path))
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        for fmt in fmts:
            if fmt == "csv":
                p = base_report + ".csv"
                _write_report_csv(results, p, elapsed_total)
                outputs_to_write.append(("CSV", p))
            else:
                p = base_report + ".json"
                _write_report_json(results, p, elapsed_total)
                outputs_to_write.append(("JSON", p))
    elif not no_auto_report:
        first_dir = output_dir
        if not first_dir:
            for r in results:
                vp = r.get("video_path")
                if vp:
                    first_dir = os.path.dirname(os.path.abspath(vp))
                    break
        if not first_dir:
            first_dir = "."
        os.makedirs(first_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_p = os.path.join(first_dir, f"pipeline_report_{stamp}.json")
        _write_report_json(results, json_p, elapsed_total)
        outputs_to_write.append(("JSON (auto)", json_p))
        csv_p = os.path.join(first_dir, f"pipeline_report_{stamp}.csv")
        _write_report_csv(results, csv_p, elapsed_total)
        outputs_to_write.append(("CSV  (auto)", csv_p))

    if outputs_to_write:
        click.echo()
        for fmt_name, path in outputs_to_write:
            click.echo(f"  {fmt_name} report -> {path}")


@cli.command("list-presets", help="Show built-in subtitle style presets")
def cmd_list_presets():
    data = list_presets()
    for name, cfg in data.items():
        click.echo(f"[{name}]")
        for k, v in cfg.items():
            if v not in ("", None, 0, 0.0, False):
                click.echo(f"  {k}: {v}")
        click.echo()


if __name__ == "__main__":
    cli()
