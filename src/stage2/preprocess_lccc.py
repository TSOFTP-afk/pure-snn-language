#!/usr/bin/env python3
# =============================================================================
# preprocess_lccc.py - Convert LCCC-base JSON corpus to plain UTF-8 text stream
# =============================================================================
#
# Usage:
#   python preprocess_lccc.py <input_dir> <output_file>
#
# LCCC-base format (pretty-printed JSON):
#   [
#   [
#   "turn 1 text",
#   "turn 2 text",
#   "turn 3 text"
#   ],
#   [
#   "..."
#   ],
#   ...
#   ]
#
# Each string is on its own line, indented. This allows streaming line-by-line
# parsing without loading the entire 910 MB JSON into memory.
#
# Output format:
#   Single UTF-8 text file, one dialogue turn per line. Lines are separated
#   by '\n' (which stage2's text_stream.cu normalizes to space).
#
# Standard library only (json, os, sys, re) -- no third-party deps.
# =============================================================================

import json
import os
import re
import sys


# Pattern: line that is just a JSON string (with optional trailing comma)
# Matches:  "  some text  "  or  "  some text  ",
# Captures the raw JSON string (with quotes) so we can use json.loads to
# properly decode escape sequences (\n, \", \uXXXX, etc.)
STRING_LINE_RE = re.compile(r'^\s*("(?:[^"\\]|\\.)*")\s*,?\s*$')


def process_file_streaming(path, out_file, stats):
    """Stream-process one JSON file line by line, extracting strings."""
    try:
        f = open(path, 'r', encoding='utf-8')
    except UnicodeDecodeError:
        f = open(path, 'r', encoding='gbk')
    except Exception as e:
        print(f"  [skip] {path}: cannot open ({e})")
        return

    file_strings = 0
    try:
        for line in f:
            m = STRING_LINE_RE.match(line)
            if not m:
                continue
            raw_json_str = m.group(1)
            try:
                # json.loads on a quoted string returns the decoded Python str
                s = json.loads(raw_json_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(s, str) or not s.strip():
                continue
            # Normalize internal whitespace: replace \r\n\t with space
            s_clean = s.replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
            out_file.write(s_clean)
            out_file.write('\n')
            file_strings += 1
    finally:
        f.close()

    stats['strings'] += file_strings
    stats['files'] += 1
    print(f"  +{file_strings:,} strings  ({os.path.basename(path)})")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_file>")
        print(f"  e.g. {sys.argv[0]} data/LCCC-base data/lccc_base.txt")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    # Find all .json / .jsonl files
    json_exts = ('.json', '.jsonl')
    json_files = []
    for root, dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if fname.lower().endswith(json_exts):
                json_files.append(os.path.join(root, fname))

    if not json_files:
        print(f"Error: no .json/.jsonl files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s) in {input_dir}")
    for f in json_files:
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.relpath(f, input_dir)}: {size_mb:.2f} MB")
    print()

    # Stream-process all files, writing directly to output
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    stats = {'strings': 0, 'files': 0}

    print("Processing (streaming, low memory)...")
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for fpath in json_files:
            process_file_streaming(fpath, out_file, stats)

    # Final stats
    out_size = os.path.getsize(output_file)
    print()
    print(f"=== Summary ===")
    print(f"  Files processed:    {stats['files']}")
    print(f"  Strings extracted:  {stats['strings']:,}")
    print(f"  Output file:        {output_file}")
    print(f"  Output size:        {out_size:,} bytes ({out_size/1024/1024:.2f} MB)")

    # Training capacity estimate
    if out_size > 0:
        print()
        print(f"=== Stage2 training estimate (1 byte/step) ===")
        for steps in [100_000, 1_000_000, 10_000_000]:
            pct = steps / out_size * 100
            coverage = "no loop" if pct < 100 else f"loops {pct/100:.1f}x"
            print(f"  {steps:>10,} steps -> {steps:>10,} bytes "
                  f"({pct:>6.2f}% of corpus, {coverage})")


if __name__ == '__main__':
    main()
