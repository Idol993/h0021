import os
import sys
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import click
from tqdm import tqdm

from presets import PRESETS, list_presets, get_preset
from merge_utils import merge_bilingual_srt, check_alignment, TIMESTAMP_TOLERANCE_MS
from caption_engine import (
    SUPPORTED_VIDEO_EXTS,
    transcribe_video,
    translate_srt,
    burn_subtitles,
    logger as engine_logger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("caption_pipeline")


def _file_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _derive_paths(video_path: str, out_dir: Optional[str] = None) -> Dict[str, str]:
    video_dir = os.path.dirname(os.path.abspath(video_path))
    stem = Path(video_path).stem
    base_dir = out_dir if out_dir else video_dir
    os.makedirs(base_dir, exist_ok=True)
    return {
        "video": video_path,
        "src_srt": os.path.join(base_dir, f"{stem}.zh.srt"),
        "tgt_srt": os.path.join(base_dir, f"{stem}.en.srt"),
        "bilingual_srt": os.path.join(base_dir, f"{stem}.bilingual.srt"),
        "output_video": os.path.join(base_dir, f"{stem}_subtitled.mp4"),
        "merge_conflict": os.path.join(base_dir, f"{stem}.merge_conflicts.log"),
        "translate_fail": os.path.join(base_dir, f"{stem}.translation_fails.log"),
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
    result = {
        "video": video_path,
        "stages": {},
        "errors": [],
        "skipped": [],
    }
    t_start = time.time()

    try:
        if not skip_transcribe:
            if _file_ok(paths["src_srt"]):
                result["skipped"].append("transcribe (already exists)")
            else:
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 1/4: Transcribe")
                t0 = time.time()
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
                result["stages"]["transcribe"] = round(time.time() - t0, 2)

        if not skip_translate:
            if _file_ok(paths["tgt_srt"]):
                result["skipped"].append("translate (already exists)")
            elif _file_ok(paths["src_srt"]):
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 2/4: Translate")
                t0 = time.time()
                translate_srt(
                    input_srt_path=paths["src_srt"],
                    output_srt_path=paths["tgt_srt"],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    fail_log_path=paths["translate_fail"],
                )
                result["stages"]["translate"] = round(time.time() - t0, 2)
            else:
                result["errors"].append("translate skipped: source SRT missing")

        if not skip_merge:
            if _file_ok(paths["bilingual_srt"]):
                result["skipped"].append("merge (already exists)")
            elif _file_ok(paths["src_srt"]) and _file_ok(paths["tgt_srt"]):
                engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 3/4: Merge Bilingual")
                t0 = time.time()
                ok, conflicts = merge_bilingual_srt(
                    src_srt_path=paths["src_srt"],
                    tgt_srt_path=paths["tgt_srt"],
                    output_path=paths["bilingual_srt"],
                    layout=merge_layout,
                    top_first=merge_top_first,
                    force=force_merge,
                    conflict_log_path=paths["merge_conflict"],
                )
                result["stages"]["merge"] = round(time.time() - t0, 2)
                if conflicts:
                    result["skipped"].append(f"merge: {len(conflicts)} conflict(s) logged")
                if not ok:
                    result["errors"].append("merge aborted due to conflicts")
            else:
                result["errors"].append("merge skipped: bilingual SRT inputs missing")

        if not skip_burn:
            if _file_ok(paths["output_video"]):
                result["skipped"].append("burn (already exists)")
            else:
                use_bilingual = _file_ok(paths["bilingual_srt"])
                use_src = _file_ok(paths["src_srt"])
                use_tgt = _file_ok(paths["tgt_srt"])
                if not (use_bilingual or use_src or use_tgt):
                    result["errors"].append("burn aborted: no SRT files available")
                else:
                    engine_logger.info(f"[{os.path.basename(video_path)}] STAGE 4/4: Burn Subtitles")
                    t0 = time.time()
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
                    result["stages"]["burn"] = round(time.time() - t0, 2)
    except Exception as e:
        result["errors"].append(f"FATAL: {str(e)}")
        result["traceback"] = traceback.format_exc()

    result["total_time"] = round(time.time() - t_start, 2)
    return result


@click.group(invoke_without_command=True)
@click.version_option(version="1.0.0", prog_name="caption-pipeline")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        engine_logger.setLevel(logging.DEBUG)


@cli.command("transcribe", help="Extract audio from video and generate SRT via faster-whisper")
@click.argument("video_path", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", "output_srt", type=click.Path(dir_okay=False), default=None, help="Output SRT path (default: <video>.zh.srt)")
@click.option("--whisper-model", type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]), default="base", help="Whisper model size")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="Compute device")
@click.option("--compute-type", type=click.Choice(["auto", "int8", "int8_float32", "int8_float16", "float16", "float32"]), default="auto", help="Compute precision")
@click.option("--language", default=None, help="Source language code (auto-detect if omitted)")
@click.option("--beam-size", type=int, default=5, help="Beam search size")
@click.option("--no-vad-filter", is_flag=True, help="Disable VAD pre-filtering")
@click.option("--no-postprocess", is_flag=True, help="Skip text post-processing (punctuation/filler/dedup)")
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
@click.option("--src-srt", default=None, help="Source-language SRT path (e.g. Chinese)")
@click.option("--tgt-srt", default=None, help="Target-language SRT path (e.g. English)")
@click.option("--bilingual-srt", default=None, help="Pre-merged bilingual SRT (takes precedence)")
@click.option("--preset", type=click.Choice(list(PRESETS.keys())), default="white", help="Built-in style preset")
@click.option("--style-json", default=None, help="Custom style JSON file path")
@click.option("--font", "font_override", default=None, help="Override font file path")
@click.option("--font-name", default=None, help="Override font name (for ASS)")
@click.option("--font-size", type=int, default=None, help="Override font size")
@click.option("--color", "primary_color_hex", default=None, help="Text color #RRGGBB")
@click.option("--outline", "outline_width", type=int, default=None, help="Outline thickness")
@click.option("--margin", "margin_v", type=int, default=None, help="Bottom margin (px)")
@click.option("--crf", type=int, default=23, help="x264 CRF (default 23, lower=better)")
@click.option("--ffmpeg-preset", type=click.Choice(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]), default="ultrafast", help="Encoding preset (speed/size tradeoff)")
def cmd_burn(video_path, output_video, src_srt, tgt_srt, bilingual_srt, preset, style_json,
             font_override, font_name, font_size, primary_color_hex, outline_width, margin_v,
             crf, ffmpeg_preset):
    if output_video is None:
        stem = Path(video_path).stem
        output_video = os.path.join(os.path.dirname(os.path.abspath(video_path)), f"{stem}_subtitled.mp4")

    overrides: Dict[str, Any] = {}
    if font_override:
        overrides["font_path"] = font_override
    if font_name:
        overrides["font_name"] = font_name
    if font_size is not None:
        overrides["font_size"] = font_size
    if outline_width is not None:
        overrides["outline_width"] = outline_width
    if margin_v is not None:
        overrides["margin_v"] = margin_v

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


@cli.command("merge", help="Merge source + target SRT into bilingual SRT")
@click.option("--src", "src_srt", type=click.Path(exists=True, dir_okay=False), required=True, help="Source language SRT")
@click.option("--tgt", "tgt_srt", type=click.Path(exists=True, dir_okay=False), required=True, help="Target language SRT")
@click.option("-o", "--output", "output_srt", type=click.Path(dir_okay=False), required=True, help="Output bilingual SRT path")
@click.option("--layout", type=click.Choice(["stack", "alternate", "source_only", "target_only"]), default="stack", help="Layout mode")
@click.option("--top", type=click.Choice(["source", "target"]), default="source", help="Which language on top (stack)")
@click.option("--force", is_flag=True, help="Force merge even with large time-axis drift")
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
        click.echo("Merge aborted due to conflicts (use --force to override)")
        sys.exit(1)
    click.echo(f"Merged -> {output_srt}")


@cli.command("check-align", help="Check time-axis alignment between two SRTs")
@click.option("--a", "a_srt", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--b", "b_srt", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--tolerance-ms", type=float, default=TIMESTAMP_TOLERANCE_MS, help="Drift tolerance in ms")
def cmd_check_align(a_srt, b_srt, tolerance_ms):
    ok, conflicts = check_alignment(a_srt, b_srt, tolerance_ms=tolerance_ms)
    click.echo(f"Aligned: {'YES' if ok else 'NO'}  ({len(conflicts)} conflict(s) beyond {tolerance_ms}ms)")
    for c in conflicts[:20]:
        click.echo(f"  Line {c.index}: start={c.start_diff_ms:.0f}ms end={c.end_diff_ms:.0f}ms")
    if len(conflicts) > 20:
        click.echo(f"  ... and {len(conflicts) - 20} more")


@cli.command("pipe", help="End-to-end pipeline: transcribe -> translate -> merge -> burn")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default=None, help="Output directory (default: same as each video)")
@click.option("--recursive", is_flag=True, help="Recurse subdirectories")
@click.option("--whisper-model", type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]), default="small", help="Whisper model size")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="Compute device")
@click.option("--compute-type", type=click.Choice(["auto", "int8", "int8_float32", "int8_float16", "float16", "float32"]), default="auto", help="Compute precision")
@click.option("--whisper-language", default=None, help="Force Whisper language (default: auto-detect)")
@click.option("--source-lang", default="zh-CN", help="Source language for translation (default zh-CN)")
@click.option("--target-lang", default="en", help="Target language for translation (default en)")
@click.option("--preset", "preset_name", type=click.Choice(list(PRESETS.keys())), default="white", help="Burn-in style preset")
@click.option("--style-json", default=None, help="Custom style JSON for burn-in")
@click.option("--merge-layout", type=click.Choice(["stack", "alternate", "source_only", "target_only"]), default="stack", help="Bilingual layout")
@click.option("--merge-top", type=click.Choice(["source", "target"]), default="source", help="Top language in stack layout")
@click.option("--force-merge", is_flag=True, help="Force merge despite time-axis drift")
@click.option("--skip-transcribe", is_flag=True, help="Skip transcribe stage")
@click.option("--skip-translate", is_flag=True, help="Skip translate stage")
@click.option("--skip-merge", is_flag=True, help="Skip merge stage")
@click.option("--skip-burn", is_flag=True, help="Skip burn stage")
@click.option("--workers", type=int, default=1, help="Parallel workers (for Whisper CPU / GPU batch)")
@click.option("--dry-run", is_flag=True, help="List videos and planned actions, exit without processing")
@click.option("--crf", type=int, default=23, help="x264 CRF for burn stage")
@click.option("--ffmpeg-preset", type=click.Choice(["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]), default="ultrafast", help="Burn encoding preset")
@click.option("--extra-margin", type=int, default=0, help="Extra bottom margin (px) for burn")
@click.option("--beam-size", type=int, default=5, help="Whisper beam size")
@click.option("--no-vad-filter", is_flag=True, help="Disable Whisper VAD filter")
def cmd_pipe(input_path, output_dir, recursive, whisper_model, device, compute_type,
             whisper_language, source_lang, target_lang, preset_name, style_json,
             merge_layout, merge_top, force_merge, skip_transcribe, skip_translate,
             skip_merge, skip_burn, workers, dry_run, crf, ffmpeg_preset, extra_margin,
             beam_size, no_vad_filter):
    videos = _collect_videos(input_path, recursive=recursive)
    if not videos:
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

    overrides: Dict[str, Any] = {}

    pending: List[str] = []
    skip_reasons: Dict[str, str] = {}

    for vp in videos:
        paths = _derive_paths(vp, output_dir)
        if not skip_transcribe and _file_ok(paths["src_srt"]) and _file_ok(paths["tgt_srt"]) \
                and _file_ok(paths["bilingual_srt"]) and _file_ok(paths["output_video"]):
            skip_reasons[vp] = "all stages done"
            continue

        if skip_transcribe:
            if not _file_ok(paths["src_srt"]):
                skip_reasons[vp] = "skip-transcribe set but src SRT missing"
                continue

        if skip_burn and not _file_ok(paths["output_video"]):
            pass

        pending.append(vp)

    click.echo(f"Found {len(videos)} video(s).")
    if skip_reasons:
        click.echo(f"Pre-skipped {len(skip_reasons)} video(s).")
    click.echo(f"Will process {len(pending)} video(s) with {workers} worker(s).\n")

    if dry_run:
        for vp in videos:
            paths = _derive_paths(vp, output_dir)
            stages = []
            if not skip_transcribe:
                stages.append("SKIP" if _file_ok(paths["src_srt"]) else "TRANSCRIBE")
            if not skip_translate:
                stages.append("SKIP" if _file_ok(paths["tgt_srt"]) else "TRANSLATE")
            if not skip_merge:
                stages.append("SKIP" if _file_ok(paths["bilingual_srt"]) else "MERGE")
            if not skip_burn:
                stages.append("SKIP" if _file_ok(paths["output_video"]) else "BURN")
            click.echo(f"  {os.path.basename(vp)}: {' -> '.join(stages)}")
        return

    task_args_list = []
    for vp in pending:
        task_args_list.append((
            vp,
            output_dir,
            whisper_model,
            device,
            compute_type,
            whisper_language,
            source_lang,
            target_lang,
            preset_name,
            style_json,
            overrides if overrides else None,
            skip_transcribe,
            skip_translate,
            skip_merge,
            skip_burn,
            force_merge,
            merge_layout,
            merge_top,
            crf,
            ffmpeg_preset,
            extra_margin,
            beam_size,
            not no_vad_filter,
        ))

    results: List[Dict[str, Any]] = []
    t_start_all = time.time()

    pbar = tqdm(total=len(task_args_list), desc="Pipeline", unit="video")

    def _on_complete(res: Dict[str, Any]):
        results.append(res)
        base = os.path.basename(res["video"])
        errors = res.get("errors", [])
        stages = res.get("stages", {})
        total = res.get("total_time", 0)
        status = "OK" if not errors else "FAIL"
        pbar.set_postfix_str(f"last:{base[:30]} {status} {total:.1f}s")
        pbar.update(1)
        if errors:
            logger.error(f"{base}: {'; '.join(errors)}")

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
                    r = {"video": a[0], "errors": [f"worker crash: {e}"], "stages": {}, "skipped": []}
                _on_complete(r)

    pbar.close()
    elapsed_total = time.time() - t_start_all

    n_ok = sum(1 for r in results if not r.get("errors"))
    n_fail = sum(1 for r in results if r.get("errors"))

    click.echo()
    click.echo("=" * 60)
    click.echo(f"Pipeline done in {elapsed_total:.1f}s.")
    click.echo(f"  OK:   {n_ok}")
    click.echo(f"  FAIL: {n_fail}")
    click.echo()

    for r in results:
        base = os.path.basename(r["video"])
        tag = "OK  " if not r.get("errors") else "FAIL"
        stages = r.get("stages", {})
        stage_str = " ".join(f"{k}={v}s" for k, v in stages.items())
        skipped = r.get("skipped", [])
        skip_str = f" (skip: {len(skipped)})" if skipped else ""
        click.echo(f"  [{tag}] {base}  total={r.get('total_time')}s  {stage_str}{skip_str}")
        for err in r.get("errors", []):
            click.echo(f"          ! {err}")


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
