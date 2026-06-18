from __future__ import annotations

import os
import re
import sys
import json
import wave
import logging
import tempfile
import subprocess
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING
from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    import srt
except ImportError:
    srt = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

if TYPE_CHECKING:
    from srt import Subtitle

from presets import SubtitleStyle, get_preset, load_style_from_json, resolve_font_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FILLER_WORDS = [
    "嗯", "啊", "那个", "这个", "就是", "然后", "其实", "对吧", "哦", "呢",
    "呃", "哈", "诶", "咦", "哟", "嘛", "啦", "咯", "哇", "诶",
    "ah", "uh", "um", "er", "oh", "like", "you know", "well", "so",
]

SUPPORTED_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm", ".wmv", ".m4v"}


def _chinese_punctuation_normalize(text: str) -> str:
    mapping = {
        ",": "，",
        "!": "！",
        "?": "？",
        ";": "；",
        ":": "：",
        "(": "（",
        ")": "）",
        "[": "【",
        "]": "】",
    }
    for eng, zh in mapping.items():
        if re.search(r"[\u4e00-\u9fff]", text):
            text = re.sub(rf"(?<=[\u4e00-\u9fff]){re.escape(eng)}(?=[\u4e00-\u9fff\s]|$)", zh, text)
            text = re.sub(rf"(^|[\s\u4e00-\u9fff]){re.escape(eng)}(?=[\u4e00-\u9fff])", zh, text)
    return text


def _remove_filler_words(text: str) -> str:
    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text))
    result = text

    if has_chinese:
        for filler in FILLER_WORDS[:20]:
            pattern = r"(^|(?<=[，。！？、；：\s]))" + re.escape(filler) + r"(?=[，。！？、；：\s]|$)"
            result = re.sub(pattern, "", result)
    else:
        lower = result.lower()
        for filler in FILLER_WORDS[20:]:
            pattern = rf"\b{re.escape(filler)}\b"
            lower = re.sub(pattern, "", lower, flags=re.IGNORECASE)
        result = " ".join(lower.split())

    result = re.sub(r"\s+", " ", result).strip()
    return result


def _remove_duplicate_segments(segments: List[Dict], max_duration_gap: float = 0.15) -> List[Dict]:
    if not segments:
        return segments

    filtered: List[Dict] = []
    prev_text = ""
    prev_end = 0.0

    for seg in segments:
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        if not text:
            continue

        start_gap = start - prev_end if prev_end > 0 else float("inf")

        if (
            prev_text
            and len(text) >= 3
            and (
                text == prev_text
                or text.startswith(prev_text + " ")
                or prev_text.endswith(text)
                or (
                    len(text) > len(prev_text) * 0.8
                    and text[: len(prev_text)] == prev_text
                    and start_gap < max_duration_gap
                )
            )
            and start_gap < max_duration_gap
        ):
            if filtered and (len(text) > len(prev_text)):
                filtered[-1]["text"] = text
                filtered[-1]["end"] = end
            continue

        filtered.append(seg)
        prev_text = text
        prev_end = end

    return filtered


def postprocess_text(text: str) -> str:
    if not text:
        return ""
    text = _chinese_punctuation_normalize(text)
    text = _remove_filler_words(text)
    return text.strip()


def probe_audio_info(video_path: str) -> Dict[str, Any]:
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is required. Install with: pip install ffmpeg-python")
    try:
        probe = ffmpeg.probe(video_path)
    except Exception as e:
        raise RuntimeError(f"ffprobe failed on {video_path}: {e}")

    audio_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise RuntimeError(f"No audio stream found in {video_path}")

    stream = audio_streams[0]
    info = {
        "sample_rate": int(stream.get("sample_rate", 44100)),
        "channels": int(stream.get("channels", 1)),
        "duration": float(probe.get("format", {}).get("duration", 0.0)),
        "codec": stream.get("codec_name", ""),
    }
    return info


def extract_audio_to_wav(video_path: str, output_wav_path: str, sample_rate: int = 16000) -> str:
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is required. Install with: pip install ffmpeg-python")

    cmd = (
        ffmpeg
        .input(video_path)
        .output(
            output_wav_path,
            ac=1,
            ar=sample_rate,
            f="wav",
            acodec="pcm_s16le",
            map="0:a:0",
        )
        .overwrite_output()
        .compile()
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr[-500:]}")
    return output_wav_path


def transcribe_video(
    video_path: str,
    output_srt_path: str,
    model_size: str = "base",
    device: str = "auto",
    compute_type: str = "auto",
    language: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = True,
    postprocess: bool = True,
) -> List[Dict]:
    if WhisperModel is None:
        raise ImportError("faster-whisper is required. Install with: pip install faster-whisper")
    if srt is None:
        raise ImportError("srt package is required. Install with: pip install srt")

    logger.info(f"[Transcribe] Processing: {os.path.basename(video_path)}")

    audio_info = probe_audio_info(video_path)
    logger.info(
        f"  Audio: {audio_info['sample_rate']}Hz, {audio_info['channels']}ch, "
        f"{audio_info['duration']:.1f}s, codec={audio_info['codec']}"
    )

    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" or device == "auto" else "float16"

    with tempfile.TemporaryDirectory(prefix="whisper_audio_") as tmpdir:
        wav_path = os.path.join(tmpdir, "audio.wav")
        extract_audio_to_wav(video_path, wav_path, sample_rate=16000)

        logger.info(f"  Loading Whisper {model_size} model ({device}, {compute_type})...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        logger.info("  Running transcription...")
        segments_iter, info = model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 500} if vad_filter else None,
        )
        logger.info(f"  Detected language: {info.language} (prob={info.language_probability:.2f})")

        raw_segments = []
        for seg in segments_iter:
            raw_segments.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text,
            })

    if postprocess:
        processed_segments = _remove_duplicate_segments(raw_segments)
        for seg in processed_segments:
            seg["text"] = postprocess_text(seg["text"])
        processed_segments = [s for s in processed_segments if s["text"]]
    else:
        processed_segments = [s for s in raw_segments if s["text"].strip()]

    srt_subs = []
    for i, seg in enumerate(processed_segments):
        srt_subs.append(
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=seg["start"]),
                end=timedelta(seconds=seg["end"]),
                content=seg["text"].strip(),
            )
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)) or ".", exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(srt_subs))

    logger.info(f"  Wrote {len(srt_subs)} segments to {output_srt_path}")
    return processed_segments


def translate_srt(
    input_srt_path: str,
    output_srt_path: str,
    source_lang: str = "zh-CN",
    target_lang: str = "en",
    fail_log_path: Optional[str] = None,
    max_chunk_len: int = 4500,
    batch_mode: bool = True,
) -> int:
    if srt is None:
        raise ImportError("srt package is required. Install with: pip install srt")
    if GoogleTranslator is None:
        raise ImportError("deep-translator is required. Install with: pip install deep-translator")

    logger.info(
        f"[Translate] {os.path.basename(input_srt_path)} : {source_lang} -> {target_lang}"
    )

    with open(input_srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    logger.info(f"  Total subtitle lines: {len(subs)}")

    if fail_log_path is None:
        fail_log_path = os.path.join(
            os.path.dirname(os.path.abspath(input_srt_path)) or ".",
            "translation_fails.log",
        )

    def _map_lang(code: str) -> str:
        mapping = {
            "zh": "zh-CN",
            "cn": "zh-CN",
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
            "en": "en",
            "en-us": "en",
            "ja": "ja",
            "jp": "ja",
            "ko": "ko",
            "kr": "ko",
            "fr": "fr",
            "de": "de",
            "es": "es",
        }
        return mapping.get(code.lower(), code)

    src = _map_lang(source_lang)
    tgt = _map_lang(target_lang)

    try:
        translator = GoogleTranslator(source=src, target=tgt)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize translator: {e}")

    translated_subs: List[Subtitle] = []
    fail_lines: List[Dict] = []

    if batch_mode and len(subs) > 1:
        chunks: List[List[int]] = []
        current_chunk: List[int] = []
        current_len = 0

        for idx, sub in enumerate(subs):
            text_len = len(sub.content) + 3
            if current_len + text_len > max_chunk_len and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [idx]
                current_len = text_len
            else:
                current_chunk.append(idx)
                current_len += text_len
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"  Batch translation mode: {len(chunks)} chunk(s)")

        for chunk_idx, chunk_indices in enumerate(chunks):
            if not chunk_indices:
                continue
            sep = "\n|||\n"
            batch_text = sep.join(subs[i].content for i in chunk_indices)

            try:
                translated = translator.translate(batch_text)
                if not translated:
                    raise ValueError("Empty translation result")
                parts = translated.split("|||")
                if len(parts) != len(chunk_indices):
                    logger.warning(
                        f"  Chunk {chunk_idx + 1}: got {len(parts)} parts, "
                        f"expected {len(chunk_indices)}. Falling back to per-line."
                    )
                    for si, orig_idx in enumerate(chunk_indices):
                        sub = subs[orig_idx]
                        try:
                            t = translator.translate(sub.content)
                            new_sub = srt.Subtitle(
                                index=len(translated_subs) + 1,
                                start=sub.start,
                                end=sub.end,
                                content=t or sub.content,
                            )
                            translated_subs.append(new_sub)
                            if not t:
                                fail_lines.append({"line": sub.index, "original": sub.content, "error": "Empty result"})
                        except Exception as e2:
                            fail_lines.append({"line": sub.index, "original": sub.content, "error": str(e2)})
                            translated_subs.append(
                                srt.Subtitle(
                                    index=len(translated_subs) + 1,
                                    start=sub.start,
                                    end=sub.end,
                                    content=sub.content,
                                )
                            )
                else:
                    for si, orig_idx in enumerate(chunk_indices):
                        sub = subs[orig_idx]
                        translated_text = parts[si].strip()
                        if not translated_text:
                            translated_text = sub.content
                            fail_lines.append({"line": sub.index, "original": sub.content, "error": "Empty in batch"})
                        translated_subs.append(
                            srt.Subtitle(
                                index=len(translated_subs) + 1,
                                start=sub.start,
                                end=sub.end,
                                content=translated_text,
                            )
                        )
            except Exception as e:
                logger.warning(f"  Chunk {chunk_idx + 1} failed ({e}), per-line fallback.")
                for orig_idx in chunk_indices:
                    sub = subs[orig_idx]
                    try:
                        t = translator.translate(sub.content)
                        new_sub = srt.Subtitle(
                            index=len(translated_subs) + 1,
                            start=sub.start,
                            end=sub.end,
                            content=t or sub.content,
                        )
                        translated_subs.append(new_sub)
                        if not t:
                            fail_lines.append({"line": sub.index, "original": sub.content, "error": "Empty"})
                    except Exception as e2:
                        fail_lines.append({"line": sub.index, "original": sub.content, "error": str(e2)})
                        translated_subs.append(
                            srt.Subtitle(
                                index=len(translated_subs) + 1,
                                start=sub.start,
                                end=sub.end,
                                content=sub.content,
                            )
                        )
    else:
        for sub in subs:
            try:
                translated_text = translator.translate(sub.content)
                if not translated_text:
                    translated_text = sub.content
                    fail_lines.append({"line": sub.index, "original": sub.content, "error": "Empty result"})
            except Exception as e:
                fail_lines.append({"line": sub.index, "original": sub.content, "error": str(e)})
                translated_text = sub.content

            translated_subs.append(
                srt.Subtitle(
                    index=len(translated_subs) + 1,
                    start=sub.start,
                    end=sub.end,
                    content=translated_text,
                )
            )

    os.makedirs(os.path.dirname(os.path.abspath(output_srt_path)) or ".", exist_ok=True)
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(translated_subs))

    if fail_lines:
        with open(fail_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {input_srt_path} ({src}->{tgt}) ===\n")
            for fl in fail_lines:
                f.write(f"[Line {fl['line']}] {fl['error']}: {fl['original'][:80]}\n")
        logger.warning(f"  Translation failures: {len(fail_lines)} (logged to {fail_log_path})")
    else:
        logger.info(f"  Translation completed successfully.")

    return len(translated_subs)


def _srt_to_ass_text(srt_path: str, style: SubtitleStyle, is_second: bool = False) -> str:
    if srt is None:
        raise ImportError("srt package is required. Install with: pip install srt")

    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    font_name = style.font_name or "Arial"
    font_size = int(style.font_size * (style.second_font_size_scale if is_second else 1.0))
    primary = style.primary_color
    outline = style.outline_color
    outline_w = style.outline_width
    shadow = style.shadow

    def fmt_time(td: timedelta) -> str:
        total_sec = td.total_seconds()
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        s = total_sec % 60
        return f"{h:d}:{m:02d}:{s:05.2f}"

    margin_v = style.margin_v
    if is_second:
        if style.background_bar:
            margin_v = margin_v + font_size + 10
        else:
            margin_v = margin_v + font_size + 8

    lines: List[str] = []
    lines.append("[Script Info]")
    lines.append("Title: Bilingual Subtitles")
    lines.append("ScriptType: v4.00+")
    lines.append("WrapStyle: 2")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append(
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    style_name = "Second" if is_second else "Default"
    lines.append(
        f"Style: {style_name},{font_name},{font_size},{primary},&H000000FF,{outline},"
        f"{style.back_color},0,0,0,0,100,100,{style.spacing},0,{style.border_style},"
        f"{outline_w},{shadow},{style.alignment},{style.margin_l},{style.margin_r},{margin_v},1"
    )
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for sub in subs:
        content = sub.content.replace("\r\n", "\\N").replace("\n", "\\N")
        content = re.sub(r"\{\\[^}]*\}", "", content)
        lines.append(
            f"Dialogue: {1 if is_second else 0},{fmt_time(sub.start)},{fmt_time(sub.end)},"
            f"{style_name},,0,0,0,,{content}"
        )

    return "\n".join(lines) + "\n"


def burn_subtitles(
    input_video_path: str,
    output_video_path: str,
    src_srt_path: Optional[str] = None,
    tgt_srt_path: Optional[str] = None,
    bilingual_srt_path: Optional[str] = None,
    style: Optional[SubtitleStyle] = None,
    preset_name: str = "white",
    custom_style_json: Optional[str] = None,
    preset_overrides: Optional[Dict[str, Any]] = None,
    crf: int = 23,
    preset: str = "ultrafast",
    audio_bitrate: str = "192k",
    extra_margin_v: int = 0,
    second_srt_on_top: bool = False,
) -> str:
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is required. Install with: pip install ffmpeg-python")

    if not bilingual_srt_path and not src_srt_path and not tgt_srt_path:
        raise ValueError("At least one SRT file must be provided (src, tgt, or bilingual)")

    if style is None:
        if custom_style_json:
            style = load_style_from_json(custom_style_json)
        else:
            style = get_preset(preset_name)

    if preset_overrides:
        for k, v in preset_overrides.items():
            if hasattr(style, k):
                setattr(style, k, v)

    if extra_margin_v:
        style.margin_v += extra_margin_v

    style.font_name = style.font_name or "Arial"
    if not style.font_path:
        resolved = resolve_font_path()
        if resolved:
            style.font_path = resolved

    logger.info(f"[Burn] {os.path.basename(input_video_path)}")
    logger.info(f"  Style: {preset_name}, font={style.font_name}({style.font_size})")

    with tempfile.TemporaryDirectory(prefix="ass_subs_") as tmpdir:
        ass_inputs: List[str] = []
        srt_paths = []
        if bilingual_srt_path:
            srt_paths.append((bilingual_srt_path, False))
        else:
            if src_srt_path:
                srt_paths.append((src_srt_path, False))
            if tgt_srt_path:
                srt_paths.append((tgt_srt_path, True))

        for i, (srt_p, is_second) in enumerate(srt_paths):
            if second_srt_on_top and is_second:
                pass
            ass_text = _srt_to_ass_text(srt_p, style, is_second=is_second)
            ass_path = os.path.join(tmpdir, f"sub{i}.ass")
            with open(ass_path, "w", encoding="utf-8") as f:
                f.write(ass_text)
            ass_inputs.append(ass_path)

        stream = ffmpeg.input(input_video_path)
        video_stream = stream["v"]
        audio_stream = stream["a"]

        font_dir = os.path.dirname(style.font_path) if style.font_path and os.path.dirname(style.font_path) else None

        for idx, ass_path in enumerate(ass_inputs):
            vf_kwargs = {"filename": ass_path}
            if font_dir:
                vf_kwargs["fontsdir"] = font_dir
            if idx == 0:
                video_stream = video_stream.filter("ass", **vf_kwargs)
            else:
                video_stream = video_stream.filter("ass", **vf_kwargs)

        output_kwargs = {
            "c:v": "libx264",
            "preset": preset,
            "crf": crf,
            "pix_fmt": "yuv420p",
            "movflags": "+faststart",
        }
        if audio_stream is not None:
            output_kwargs["c:a"] = "aac"
            output_kwargs["b:a"] = audio_bitrate

        os.makedirs(os.path.dirname(os.path.abspath(output_video_path)) or ".", exist_ok=True)
        stream_out = ffmpeg.output(video_stream, audio_stream, output_video_path, **output_kwargs)
        cmd = stream_out.overwrite_output().compile()

        logger.info(f"  Running FFmpeg (preset={preset}, crf={crf})...")
        logger.debug(f"  CMD: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg burn failed: {result.stderr[-800:]}")

    logger.info(f"  Output: {output_video_path}")
    return output_video_path
