"""Global ID generator for Propositions and Chunks."""

import threading


class IDGenerator:
    """Thread-safe global ID generator for chunks and propositions."""

    def __init__(self):
        self._chunk_counter = 0
        self._prop_counter = 0
        self._lock = threading.Lock()

    def set_counters(self, chunk: int = 0, prop: int = 0):
        """Set the chunk and proposition counters to specific values."""
        with self._lock:
            self._chunk_counter = chunk
            self._prop_counter = prop

    def next_chunk_id(self) -> str:
        """Generate next unique chunk ID."""
        with self._lock:
            self._chunk_counter += 1
            return f"chunk_{self._chunk_counter}"

    def next_prop_id(self) -> str:
        """Generate next unique proposition ID."""
        with self._lock:
            self._prop_counter += 1
            return f"prop_{self._prop_counter}"

    def reset(self):
        """Reset all counters (useful for testing)."""
        with self._lock:
            self._chunk_counter = 0
            self._prop_counter = 0


# Global singleton instance
_id_generator = IDGenerator()

def set_counters(chunk: int = 0, prop: int = 0):
    """Set the chunk and proposition counters to specific values."""
    _id_generator.set_counters(chunk=chunk, prop=prop)


def get_next_chunk_id() -> str:
    """Get next unique chunk ID from global counter."""
    return _id_generator.next_chunk_id()


def get_next_prop_id() -> str:
    """Get next unique proposition ID from global counter."""
    return _id_generator.next_prop_id()


def reset_id_counters():
    """Reset all ID counters (useful for testing)."""
    _id_generator.reset()
