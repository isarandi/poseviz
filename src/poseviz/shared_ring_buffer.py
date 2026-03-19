import multiprocessing.sharedctypes

import numpy as np


class SharedRingBuffer:
    """Ring buffer backed by multiprocessing shared memory.

    Each slot holds a fixed-size array. Multiple processes can read/write slots
    by index without copying through pipes or pickle.

    Args:
        n_slots: Number of slots in the ring buffer.
        max_elems: Maximum number of elements per slot.
        dtype: Numpy dtype of the elements.
    """

    def __init__(self, n_slots: int, max_elems: int, dtype=np.uint8):
        self.n_slots = n_slots
        self.max_elems = max_elems
        self.dtype = np.dtype(dtype)
        ctype = np.ctypeslib.as_ctypes_type(self.dtype)
        self.max_bytes_per_slot = self.max_elems * self.dtype.itemsize
        self._raw = multiprocessing.sharedctypes.RawArray(ctype, n_slots * max_elems)

    def write(self, index: int, data: np.ndarray):
        """Write data into a slot. Returns (shape, dtype, index) descriptor."""
        dst = self._slot_array(index, data.shape)
        np.copyto(dst, data)
        return dst.shape, self.dtype, index

    def read(self, index: int, shape: tuple) -> np.ndarray:
        """Read a slot as a numpy view (zero-copy)."""
        return self._slot_array(index, shape)

    def get_slot(self, index: int, shape: tuple) -> np.ndarray:
        """Get a writable numpy view into a slot (zero-copy).

        Use this to let another function write directly into shared memory
        (e.g., as a destination array for cv2.resize).
        """
        return self._slot_array(index, shape)

    def _slot_array(self, index: int, shape: tuple) -> np.ndarray:
        n_elems = int(np.prod(shape))
        offset = index * self.max_bytes_per_slot
        return np.frombuffer(
            self._raw, dtype=self.dtype, count=n_elems, offset=offset
        ).reshape(shape)
