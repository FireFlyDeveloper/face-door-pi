"""
Activity/Audit log system for face recognition door system.
Logs events as JSON lines to a file with ISO timestamps.
"""

import json
import os
from datetime import datetime, timezone


class ActivityLogger:
    """Logs face recognition events to a JSON-lines file."""

    def __init__(self, log_path='/home/admin/face-door-system/activity.log'):
        self.log_path = log_path
        # Ensure directory exists
        log_dir = os.path.dirname(self.log_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def log_event(self, face_id, result, details=''):
        """Append a JSON line to the log file with an ISO timestamp."""
        entry = {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'face_id': face_id,
            'result': result,
            'details': details
        }
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except OSError as e:
            print(f"[ActivityLogger] Failed to write log: {e}")

    def get_logs(self, limit=50):
        """Return most recent `limit` log entries as a list of dicts."""
        entries = []
        try:
            if not os.path.exists(self.log_path):
                return []
            with open(self.log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as e:
            print(f"[ActivityLogger] Failed to read log: {e}")
            return []
        # Most recent first
        entries.reverse()
        return entries[:limit]

    def clear_logs(self):
        """Truncate the log file."""
        try:
            open(self.log_path, 'w').close()
            print(f"[ActivityLogger] Log cleared: {self.log_path}")
        except OSError as e:
            print(f"[ActivityLogger] Failed to clear log: {e}")
