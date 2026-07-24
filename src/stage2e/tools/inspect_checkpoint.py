#!/usr/bin/env python3
"""Inspect and optionally checksum a Stage 2e v3 checkpoint without CUDA."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path


MAGIC = b"SNN2ECP3"
FOOTER_MAGIC = b"SNN2EOK3"
HEADER = struct.Struct("<8sIIIIQQIIII")
SECTION = struct.Struct("<48sQ")
FOOTER = struct.Struct("<8sQ")


def crc32_stream(handle, size: int, chunk_size: int = 8 * 1024 * 1024) -> int:
    value = 0
    remaining = size
    while remaining:
        chunk = handle.read(min(chunk_size, remaining))
        if not chunk:
            raise ValueError("checkpoint payload is truncated")
        value = zlib.crc32(chunk, value)
        remaining -= len(chunk)
    return value


def inspect(path: Path, verify: bool = False) -> dict:
    with path.open("rb") as handle:
        raw = handle.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError("checkpoint header is truncated")
        (magic, version, header_bytes, section_count, _reserved,
         payload_bytes, payload_checksum, n_neurons, n_synapses,
         synapse_bytes, neuron_bytes) = HEADER.unpack(raw)
        if magic != MAGIC or version != 3 or header_bytes != HEADER.size:
            raise ValueError("not a compatible Stage 2e v3 checkpoint")

        sections = []
        section_bytes = 0
        for _ in range(section_count):
            raw = handle.read(SECTION.size)
            if len(raw) != SECTION.size:
                raise ValueError("section table is truncated")
            name, size = SECTION.unpack(raw)
            name = name.split(b"\0", 1)[0].decode("ascii")
            sections.append({"name": name, "bytes": size})
            section_bytes += size
        if section_bytes != payload_bytes:
            raise ValueError("section sizes do not match payload size")

        payload_offset = handle.tell()
        expected_size = payload_offset + payload_bytes + FOOTER.size
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"unexpected file size: {actual_size} != {expected_size}")

        verified = False
        if verify:
            checksum = crc32_stream(handle, payload_bytes)
            footer_raw = handle.read(FOOTER.size)
            footer_magic, footer_checksum = FOOTER.unpack(footer_raw)
            if footer_magic != FOOTER_MAGIC:
                raise ValueError("checkpoint completion footer is missing")
            if checksum != payload_checksum or checksum != footer_checksum:
                raise ValueError("checkpoint payload checksum mismatch")
            verified = True
        else:
            handle.seek(payload_offset + payload_bytes)
            footer_raw = handle.read(FOOTER.size)
            footer_magic, footer_checksum = FOOTER.unpack(footer_raw)
            if footer_magic != FOOTER_MAGIC or footer_checksum != payload_checksum:
                raise ValueError("checkpoint completion footer is invalid")

    return {
        "path": str(path),
        "version": version,
        "n_neurons": n_neurons,
        "n_synapses": n_synapses,
        "bio_synapse_bytes": synapse_bytes,
        "neuron_state_bytes": neuron_bytes,
        "payload_bytes": payload_bytes,
        "payload_mib": round(payload_bytes / (1024 * 1024), 2),
        "payload_checksum": f"{payload_checksum:016x}",
        "checksum_verified": verified,
        "sections": sections,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--verify", action="store_true", help="stream and verify the full payload")
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.checkpoint, args.verify), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
