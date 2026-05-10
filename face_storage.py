"""
face_storage.py — JSON-based face encoding storage for Raspberry Pi face door system.

Stores up to MAX_FACES (5) face encodings as JSON. Each encoding is a list of 10
128-D numpy arrays stored as nested lists. Provides full CRUD operations and
activity logging.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FaceStorage:
    """JSON-based persistent storage for face encodings with activity logging."""

    MAX_FACES = 5

    def __init__(
        self,
        path: str = "/home/admin/face-door-system/faces.json",
        log_path: str = "/home/admin/face-door-system/activity.log",
    ):
        self.path = path
        self.log_path = log_path
        self._faces: Dict[str, Dict[str, Any]] = {}

        # Ensure directories exist
        for p in (self.path, self.log_path):
            dirname = os.path.dirname(p)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname, exist_ok=True)

        # Set up activity logger
        self.logger = logging.getLogger("FaceDoorActivity")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Load existing faces
        self._faces = self.load_faces()

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def encode_encoding(encoding: np.ndarray) -> List[float]:
        """Convert a 128-D numpy encoding array to a JSON-serializable list."""
        if encoding.shape != (128,):
            raise ValueError(f"Encoding must be 128-D, got shape {encoding.shape}")
        return encoding.tolist()

    @staticmethod
    def decode_encoding(data: List[float]) -> np.ndarray:
        """Convert a JSON-serialized list back into a 128-D numpy array."""
        arr = np.array(data, dtype=np.float64)
        if arr.shape != (128,):
            raise ValueError(f"Decoded encoding must be 128-D, got shape {arr.shape}")
        return arr

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_faces(self) -> Dict[str, Dict[str, Any]]:
        """Load all faces from the JSON file. Returns dict of face_id -> face_data."""
        if not os.path.exists(self.path):
            self.logger.info("No existing faces file — starting fresh")
            return {}

        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            # Validate structure — each value should be a dict with 'encoding' key
            for face_id, face_data in list(data.items()):
                if not isinstance(face_data, dict) or "encoding" not in face_data:
                    self.logger.warning(f"Corrupt entry for '{face_id}' — removing")
                    del data[face_id]
                    continue
                # Ensure encoding is list of 10 lists of 128 floats
                enc_list = face_data["encoding"]
                if not isinstance(enc_list, list) or len(enc_list) != 10:
                    self.logger.warning(
                        f"Invalid encoding list length for '{face_id}' "
                        f"({len(enc_list)}), expected 10 — removing"
                    )
                    del data[face_id]
                    continue
            self._faces = data
            self.logger.info(f"Loaded {len(data)} face(s) from {self.path}")
            return data
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"Failed to load faces: {e}")
            return {}

    def save_faces(self, faces: Dict[str, Dict[str, Any]]) -> None:
        """Persist face data dict to JSON file."""
        try:
            with open(self.path, "w") as f:
                json.dump(faces, f, indent=2)
            self._faces = faces
            self.logger.info(f"Saved {len(faces)} face(s) to {self.path}")
        except (IOError, TypeError) as e:
            self.logger.error(f"Failed to save faces: {e}")
            raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_face(
        self,
        face_id: str,
        encoding_list: List[np.ndarray],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a new face.

        Args:
            face_id: Unique identifier (e.g. 'alice' or 'uuid-...').
            encoding_list: Exactly 10 np.ndarrays, each 128-D.
            metadata: Optional dict of extra info (name, timestamp, etc.).

        Returns:
            True if added, False if at capacity or duplicate.

        Raises:
            ValueError: If encoding_list is malformed.
        """
        if not isinstance(encoding_list, list) or len(encoding_list) != 10:
            raise ValueError(
                f"encoding_list must be a list of 10 arrays, got {len(encoding_list)}"
            )
        for i, enc in enumerate(encoding_list):
            if not isinstance(enc, np.ndarray) or enc.shape != (128,):
                raise ValueError(f"encoding_list[{i}] is not a 128-D numpy array")

        # Check capacity
        if face_id not in self._faces and len(self._faces) >= self.MAX_FACES:
            self.logger.warning(
                f"Cannot add '{face_id}' — at capacity ({self.MAX_FACES})"
            )
            return False

        # Build entry
        encoded_encodings = [self.encode_encoding(enc) for enc in encoding_list]
        entry: Dict[str, Any] = {"encoding": encoded_encodings}
        if metadata:
            entry["metadata"] = metadata
        entry["created_at"] = datetime.now().isoformat()

        self._faces[face_id] = entry
        self.save_faces(self._faces)
        self.logger.info(f"Added face '{face_id}' (total: {len(self._faces)})")
        return True

    def delete_face(self, face_id: str) -> bool:
        """Remove a face by ID. Returns True if it existed and was deleted."""
        if face_id not in self._faces:
            self.logger.warning(f"Face '{face_id}' not found — nothing to delete")
            return False

        del self._faces[face_id]
        self.save_faces(self._faces)
        self.logger.info(f"Deleted face '{face_id}' (remaining: {len(self._faces)})")
        return True

    def list_faces(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a copy of all stored faces with decodable encodings baked in.

        Returns:
            dict of face_id -> {'encoding': [np.ndarray, ... 10],
                                'metadata': {...}, 'created_at': str}
        """
        result: Dict[str, Dict[str, Any]] = {}
        for face_id, data in self._faces.items():
            decoded = [self.decode_encoding(e) for e in data["encoding"]]
            entry: Dict[str, Any] = {"encoding": decoded}
            if "metadata" in data:
                entry["metadata"] = data["metadata"]
            if "created_at" in data:
                entry["created_at"] = data["created_at"]
            result[face_id] = entry
        return result

    def get_face_count(self) -> int:
        """Return the number of currently stored faces."""
        return len(self._faces)

    def get_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """
        Return decoded face data for a single ID, or None if not found.
        """
        data = self._faces.get(face_id)
        if data is None:
            return None
        decoded = [self.decode_encoding(e) for e in data["encoding"]]
        entry: Dict[str, Any] = {"encoding": decoded}
        if "metadata" in data:
            entry["metadata"] = data["metadata"]
        if "created_at" in data:
            entry["created_at"] = data["created_at"]
        return entry
