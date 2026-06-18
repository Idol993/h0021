from __future__ import annotations

import os
import logging
from datetime import timedelta
from typing import List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

try:
    import srt
except ImportError:
    srt = None

if TYPE_CHECKING:
    from srt import Subtitle

logger = logging.getLogger(__name__)


@dataclass
class MergeConflict:
    index: int
    reason: str
    start_diff_ms: float = 0.0
    end_diff_ms: float = 0.0


TIMESTAMP_TOLERANCE_MS = 200.0


def parse_srt_file(srt_path: str) -> List["Subtitle"]:
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    subs = list(srt.parse(content))
    subs.sort(key=lambda x: x.start)
    return subs


def format_timestamp(td: timedelta) -> str:
    total_seconds = int(td.total_seconds())
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    microseconds = td.microseconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{microseconds:03d}"


def merge_bilingual_srt(
    src_srt_path: str,
    tgt_srt_path: str,
    output_path: str,
    layout: str = "stack",
    top_first: str = "source",
    force: bool = False,
    conflict_log_path: Optional[str] = None,
) -> Tuple[bool, List[MergeConflict]]:
    if srt is None:
        raise ImportError("srt package is required. Install with: pip install srt")

    src_subs = parse_srt_file(src_srt_path)
    tgt_subs = parse_srt_file(tgt_srt_path)
    conflicts: List[MergeConflict] = []

    len_src, len_tgt = len(src_subs), len(tgt_subs)
    if len_src != len_tgt:
        conflicts.append(
            MergeConflict(
                index=-1,
                reason=f"Subtitle count mismatch: source={len_src}, target={len_tgt}",
            )
        )

    max_len = max(len_src, len_tgt)
    merged_subs: List[Subtitle] = []

    for i in range(max_len):
        src_sub = src_subs[i] if i < len_src else None
        tgt_sub = tgt_subs[i] if i < len_tgt else None

        if src_sub is None:
            conflicts.append(MergeConflict(index=i + 1, reason="Source subtitle missing, using target only"))
            merged_subs.append(tgt_sub)
            continue
        if tgt_sub is None:
            conflicts.append(MergeConflict(index=i + 1, reason="Target subtitle missing, using source only"))
            merged_subs.append(src_sub)
            continue

        start_diff = abs((src_sub.start - tgt_sub.start).total_seconds()) * 1000.0
        end_diff = abs((src_sub.end - tgt_sub.end).total_seconds()) * 1000.0

        if start_diff > TIMESTAMP_TOLERANCE_MS or end_diff > TIMESTAMP_TOLERANCE_MS:
            conflicts.append(
                MergeConflict(
                    index=i + 1,
                    reason=f"Timestamp drift exceeds {TIMESTAMP_TOLERANCE_MS}ms",
                    start_diff_ms=start_diff,
                    end_diff_ms=end_diff,
                )
            )

        if start_diff > TIMESTAMP_TOLERANCE_MS * 3 or end_diff > TIMESTAMP_TOLERANCE_MS * 3:
            if not force:
                return False, conflicts

        final_start = src_sub.start
        final_end = src_sub.end

        if layout == "stack":
            if top_first == "source":
                top_text = src_sub.content.strip()
                bottom_text = tgt_sub.content.strip()
            else:
                top_text = tgt_sub.content.strip()
                bottom_text = src_sub.content.strip()
            combined = f"{top_text}\n{bottom_text}"
        elif layout == "alternate":
            if i % 2 == 0:
                combined = src_sub.content.strip() + "\n" + tgt_sub.content.strip()
            else:
                combined = tgt_sub.content.strip() + "\n" + src_sub.content.strip()
        elif layout == "source_only":
            combined = src_sub.content.strip()
        elif layout == "target_only":
            combined = tgt_sub.content.strip()
        else:
            combined = src_sub.content.strip() + "\n" + tgt_sub.content.strip()

        merged_subs.append(
            srt.Subtitle(
                index=len(merged_subs) + 1,
                start=final_start,
                end=final_end,
                content=combined,
            )
        )

    for idx, sub in enumerate(merged_subs):
        sub.index = idx + 1

    output_content = srt.compose(merged_subs)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    if conflict_log_path and conflicts:
        with open(conflict_log_path, "w", encoding="utf-8") as f:
            f.write(f"Bilingual SRT Merge Conflict Report\n")
            f.write(f"Source: {src_srt_path}\n")
            f.write(f"Target: {tgt_srt_path}\n")
            f.write(f"Output: {output_path}\n")
            f.write(f"Total conflicts: {len(conflicts)}\n")
            f.write("=" * 60 + "\n\n")
            for c in conflicts:
                f.write(f"[Line {c.index}] {c.reason}\n")
                if c.start_diff_ms or c.end_diff_ms:
                    f.write(f"  Start diff: {c.start_diff_ms:.1f}ms, End diff: {c.end_diff_ms:.1f}ms\n")
                f.write("\n")

    return True, conflicts


def check_alignment(
    srt_a_path: str,
    srt_b_path: str,
    tolerance_ms: float = TIMESTAMP_TOLERANCE_MS,
) -> Tuple[bool, List[MergeConflict]]:
    if srt is None:
        raise ImportError("srt package is required. Install with: pip install srt")

    subs_a = parse_srt_file(srt_a_path)
    subs_b = parse_srt_file(srt_b_path)
    conflicts: List[MergeConflict] = []

    if len(subs_a) != len(subs_b):
        conflicts.append(
            MergeConflict(
                index=-1,
                reason=f"Subtitle count mismatch: A={len(subs_a)}, B={len(subs_b)}",
            )
        )

    min_len = min(len(subs_a), len(subs_b))
    for i in range(min_len):
        a, b = subs_a[i], subs_b[i]
        start_diff = abs((a.start - b.start).total_seconds()) * 1000.0
        end_diff = abs((a.end - b.end).total_seconds()) * 1000.0
        if start_diff > tolerance_ms or end_diff > tolerance_ms:
            conflicts.append(
                MergeConflict(
                    index=i + 1,
                    reason=f"Timestamp drift exceeds {tolerance_ms}ms",
                    start_diff_ms=start_diff,
                    end_diff_ms=end_diff,
                )
            )

    is_aligned = len(conflicts) == 0
    return is_aligned, conflicts
