#!/usr/bin/env python3
"""Convert the official LCCC JSONL archive into a streaming text corpus."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="LCCC .jsonl or .jsonl.gz")
    parser.add_argument("--output", required=True, help="Plain UTF-8 output corpus")
    parser.add_argument(
        "--max-dialogs",
        type=int,
        default=0,
        help="Optional nonzero limit for smoke tests",
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def convert(input_path: Path, output_path: Path, max_dialogs: int = 0
            ) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    dialogs = 0
    utterances = 0
    try:
        with open_text(input_path) as source, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                line = line.strip()
                if not line:
                    continue
                dialog = json.loads(line)
                if not isinstance(dialog, list) or not all(
                    isinstance(item, str) for item in dialog
                ):
                    raise ValueError(
                        f"line {line_number} is not a JSON array of strings"
                    )
                cleaned = [item.strip() for item in dialog if item.strip()]
                if not cleaned:
                    continue
                destination.write("\n".join(cleaned))
                destination.write("\n\n")
                dialogs += 1
                utterances += len(cleaned)
                if max_dialogs and dialogs >= max_dialogs:
                    break
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dialogs, utterances


def main() -> None:
    args = parse_args()
    dialogs, utterances = convert(
        Path(args.input), Path(args.output), args.max_dialogs
    )
    size = Path(args.output).stat().st_size
    print(
        f"PREPARED dialogs={dialogs} utterances={utterances} "
        f"bytes={size} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
