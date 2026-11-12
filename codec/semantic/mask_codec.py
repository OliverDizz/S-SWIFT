"""Lossless binary codec for the semantic base (mask) and structural (edge) layers.

The masks and edges are binary 0/1 maps. They are packed with ``np.packbits``
and stored with ``np.savez_compressed`` (a lossless DEFLATE container). The
operative metric for these layers is bytes per frame.
"""

import io

import numpy as np


def _to_bytes(binary_map):
    arr = (binary_map > 0).astype(np.uint8)
    packed = np.packbits(arr.reshape(-1))
    buf = io.BytesIO()
    np.savez_compressed(buf, shape=np.array(arr.shape, dtype=np.int64),
                        codes=packed)
    return buf.getvalue()


def encode(binary_map):
    """Encode a binary map to a compressed byte string."""
    return _to_bytes(binary_map)


def decode(data, shape):
    """Decode a compressed byte string back to a binary map."""
    buf = io.BytesIO(data)
    with np.load(buf) as z:
        codes = z['codes']
        shape = tuple(z['shape'])
    arr = np.unpackbits(codes)[:shape[0] * shape[1]].reshape(shape)
    return arr.astype(np.uint8)


def bitrate_bytes(binary_map):
    """Return the compressed size of one binary map in bytes."""
    return len(_to_bytes(binary_map))


def code_bitrate_bytes(code_map):
    """DEFLATE size of a progressive code map with masked positions omitted.

    ``code_map`` is ``(iterations * bits, h, w)`` of values in ``{-1, 0, +1}``.
    A zero entry is a masked (non-transmitted) position, so only non-zero
    positions are packed, in raster order over non-masked positions. For an
    unmasked map (all values ``{-1, +1}``) this matches ``bitrate_bytes``
    byte-for-byte.
    """
    values = (code_map.astype(np.int8) + 1) // 2  # {-1, 0, +1} -> {0, 0, 1}
    keep = (code_map != 0).reshape(-1)
    packed = np.packbits(values.reshape(-1)[keep])
    buf = io.BytesIO()
    np.savez_compressed(buf, shape=np.array(code_map.shape, dtype=np.int64),
                        codes=packed)
    return len(buf.getvalue())
