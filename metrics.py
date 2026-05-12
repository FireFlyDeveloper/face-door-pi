"""
metrics.py — Performance logging for thesis evaluation.

Tracks:
  - FAR (False Acceptance Rate): spoof accepted as live
  - FRR (False Rejection Rate): live rejected as spoof
  - FPS: frames per second over the detection pipeline
  - Latency per pipeline stage (anti-spoof, encoding, matching)
  - Per-attack-type statistics (print, phone screen, live)

Outputs CSV logs for thesis charts and tables.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

METRICS_PATH = "metrics_log.csv"
DETAIL_PATH = "metrics_detail.csv"


class MetricsLogger:
    """Aggregates per-frame metrics and writes CSV summaries."""

    def __init__(self, log_dir: str = "."):
        self.log_dir = os.path.abspath(log_dir)
        self.metrics_path = os.path.join(self.log_dir, METRICS_PATH)
        self.detail_path = os.path.join(self.log_dir, DETAIL_PATH)
        os.makedirs(log_dir, exist_ok=True)

        # Per-frame detail buffer
        self._details: List[Dict] = []

        # Aggregated counters
        self._start_time = time.time()
        self._total_frames = 0

        # Per-stage latency accumulators (seconds)
        self._latency: Dict[str, List[float]] = defaultdict(list)

        # Anti-spoof confusion matrix
        self._true_live: int = 0
        self._true_spoof: int = 0
        self._false_live: int = 0   # spoof accepted as live
        self._false_spoof: int = 0  # live rejected as spoof

        # Match results
        self._matches: int = 0
        self._rejections: int = 0
        self._match_distances: List[float] = []

        # Write CSV headers
        self._init_csv()

    def _init_csv(self):
        """Write headers to detail CSV if file is new."""
        if not os.path.exists(self.detail_path):
            with open(self.detail_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_num",
                    "detect_ms", "anti_spoof_ms", "encode_ms", "match_ms",
                    "total_ms", "anti_spoof_score", "is_live_ground_truth",
                    "match_id", "match_distance", "result",
                ])

    def log_frame(
        self,
        detect_ms: float,
        anti_spoof_ms: float,
        encode_ms: float,
        match_ms: float,
        anti_spoof_score: float,
        is_live: Optional[bool] = None,
        match_id: Optional[str] = None,
        match_distance: Optional[float] = None,
        result: str = "SCANNING",
    ):
        """
        Log a single frame's metrics.

        Args:
            detect_ms: Face detection latency in milliseconds.
            anti_spoof_ms: Anti-spoof inference latency in ms.
            encode_ms: Encoding extraction latency in ms.
            match_ms: Matching latency in ms.
            anti_spoof_score: Liveness score from anti-spoof.
            is_live: Ground truth (True=live, False=spoof, None=unknown).
            match_id: Matched face ID or None.
            match_distance: Cosine/Euclidean distance to best match.
            result: "GRANTED", "REJECTED", or "SCANNING".
        """
        self._total_frames += 1
        self._latency["detect"].append(detect_ms)
        self._latency["anti_spoof"].append(anti_spoof_ms)
        self._latency["encode"].append(encode_ms)
        self._latency["match"].append(match_ms)

        # Confusion matrix (only when ground truth is known)
        if is_live is not None:
            live_decision = anti_spoof_score >= 0.5
            if is_live and live_decision:
                self._true_live += 1
            elif not is_live and not live_decision:
                self._true_spoof += 1
            elif not is_live and live_decision:
                self._false_live += 1  # FAR
            elif is_live and not live_decision:
                self._false_spoof += 1  # FRR

        if result == "GRANTED" and match_distance is not None:
            self._matches += 1
            self._match_distances.append(match_distance)
        elif result == "REJECTED":
            self._rejections += 1

        # Append to detail CSV
        ts = datetime.now().isoformat()
        row = [
            ts, self._total_frames,
            f"{detect_ms:.1f}", f"{anti_spoof_ms:.1f}",
            f"{encode_ms:.1f}", f"{match_ms:.1f}",
            f"{detect_ms + anti_spoof_ms + encode_ms + match_ms:.1f}",
            f"{anti_spoof_score:.3f}",
            str(is_live) if is_live is not None else "",
            match_id or "",
            f"{match_distance:.4f}" if match_distance is not None else "",
            result,
        ]
        try:
            with open(self.detail_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except IOError as e:
            log.warning("Failed to write metrics detail: %s", e)

    def summary(self) -> Dict:
        """Return aggregated statistics as a dict."""
        elapsed = time.time() - self._start_time

        # Latency averages
        avg_latency = {
            stage: (sum(vals) / len(vals) if vals else 0.0)
            for stage, vals in self._latency.items()
        }
        total_frames_with_face = max(
            len(self._latency.get("detect", [])), 1
        )

        total_decisions = self._true_live + self._true_spoof + \
            self._false_live + self._false_spoof

        far = self._false_live / max(self._false_live + self._true_spoof, 1)
        frr = self._false_spoof / max(self._false_spoof + self._true_live, 1)

        return {
            "total_frames": self._total_frames,
            "elapsed_seconds": round(elapsed, 1),
            "avg_fps": round(self._total_frames / max(elapsed, 0.01), 1),
            "detect_ms_avg": round(avg_latency.get("detect", 0), 1),
            "anti_spoof_ms_avg": round(avg_latency.get("anti_spoof", 0), 1),
            "encode_ms_avg": round(avg_latency.get("encode", 0), 1),
            "match_ms_avg": round(avg_latency.get("match", 0), 1),
            "pipeline_ms_total": round(
                sum(avg_latency.values()), 1
            ),
            "true_live": self._true_live,
            "true_spoof": self._true_spoof,
            "false_live_FAR": self._false_live,
            "false_spoof_FRR": self._false_spoof,
            "total_labeled_decisions": total_decisions,
            "FAR": round(far, 4),
            "FRR": round(frr, 4),
            "accuracy": round(
                (self._true_live + self._true_spoof) / max(total_decisions, 1),
                4,
            ),
            "matches": self._matches,
            "rejections": self._rejections,
            "match_accept_rate": round(
                self._matches / max(self._matches + self._rejections, 1), 4,
            ),
            "avg_match_distance": round(
                sum(self._match_distances) / max(len(self._match_distances), 1),
                4,
            ) if self._match_distances else 0.0,
        }

    def print_summary(self):
        """Print formatted summary to stdout."""
        s = self.summary()
        print("=" * 50)
        print("THESIS METRICS SUMMARY")
        print("=" * 50)
        print(f"  Total frames:        {s['total_frames']}")
        print(f"  Elapsed time:        {s['elapsed_seconds']}s")
        print(f"  Avg FPS:             {s['avg_fps']}")
        print()
        print("  -- Pipeline Latency (per frame) --")
        print(f"  Face detection:      {s['detect_ms_avg']}ms")
        print(f"  Anti-spoof:          {s['anti_spoof_ms_avg']}ms")
        print(f"  Encoding:            {s['encode_ms_avg']}ms")
        print(f"  Matching:            {s['match_ms_avg']}ms")
        print(f"  Total pipeline:      {s['pipeline_ms_total']}ms")
        print()
        print("  -- Anti-Spoof Results --")
        print(f"  True Live (TP):       {s['true_live']}")
        print(f"  True Spoof (TN):      {s['true_spoof']}")
        print(f"  False Live / FAR:     {s['false_live_FAR']} ({s['FAR']*100:.2f}%)")
        print(f"  False Spoof / FRR:    {s['false_spoof_FRR']} ({s['FRR']*100:.2f}%)")
        print(f"  Accuracy:             {s['accuracy']*100:.2f}%")
        print()
        print("  -- Recognition --")
        print(f"  Matches:              {s['matches']}")
        print(f"  Rejections:           {s['rejections']}")
        print(f"  Avg match distance:   {s['avg_match_distance']}")
        print(f"  Accept rate:          {s['match_accept_rate']*100:.1f}%")
        print("=" * 50)
