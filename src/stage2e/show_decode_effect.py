"""
解码效果可视化工具 (v3 checkpoint 版)
====================================
从新版 v3 格式 checkpoint (.snn2e) 读取 "neuron_byte_counts" section (55K×256 矩阵),
对每个 L6 神经元找 argmax 字节, 统计每个字节的偏好神经元数,
然后读取测试文本, 展示 "原始文本 vs 网络可识别字节" 的对比效果。

v3 格式 (与 scheduler_checkpoint.cu::save_checkpoint 对应):
  - 头部 56 字节 (含 magic "SNN2ECP3", version=3, section_count 等)
  - section 表 (每个 56 字节, 含 name + bytes)
  - payload (各 section 数据连续存储)
  - footer 16 字节 (magic "SNN2EOK3" + checksum)

用法:
    python show_decode_effect.py [checkpoint.snn2e] [text.txt]
"""
import struct
import sys
from pathlib import Path

# ===== v3 格式常量 =====
MAGIC = b"SNN2ECP3"
FOOTER_MAGIC = b"SNN2EOK3"
VERSION = 3
HEADER_FMT = "<8sIIIIQQIIII"   # 56 bytes
SECTION_FMT = "<48sQ"           # 56 bytes
FOOTER_FMT = "<8sQ"             # 16 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
SECTION_SIZE = struct.calcsize(SECTION_FMT)
FOOTER_SIZE = struct.calcsize(FOOTER_FMT)

# ===== 网络结构常量 (与 config.h 一致) =====
N_NEURONS = 55_000
N_COLUMNS = 50
NEURONS_PER_COLUMN = 1000
L6_OFFSET_IN_COL = 750  # L4(200) + L2/3(350) + L5(200)
L6_SIZE_PER_COL = 250
N_L6_TOTAL = N_COLUMNS * L6_SIZE_PER_COL  # 12500
L6_THRESHOLD = 5  # 与解码器默认一致

# ===== 默认路径 =====
DEFAULT_CKPT = Path("src/stage2e/checkpoints/ckpt_step50000.snn2e")
DEFAULT_TEXT = Path("data/lccc_sample_1mb.txt")


# ===== 1. v3 checkpoint 读取 =====
def open_v3_checkpoint(path):
    """打开 v3 checkpoint, 返回 (header_dict, sections_dict, file_handle)
    sections_dict: {section_name: (bytes, file_offset)}
    """
    f = open(path, "rb")
    raw = f.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        f.close()
        raise ValueError(f"checkpoint header truncated: {len(raw)} bytes")
    (magic, version, header_bytes, section_count, _reserved,
     payload_bytes, payload_checksum, n_neurons, n_synapses,
     bio_synapse_bytes, neuron_state_bytes) = struct.unpack(HEADER_FMT, raw)
    if magic != MAGIC:
        f.close()
        raise ValueError(f"magic mismatch: {magic!r} (not a v3 checkpoint?)")
    if version != VERSION:
        f.close()
        raise ValueError(f"version mismatch: {version} (expected {VERSION})")
    if header_bytes != HEADER_SIZE:
        f.close()
        raise ValueError(f"header_bytes mismatch: {header_bytes} (expected {HEADER_SIZE})")

    # 读 section 表
    sections = {}
    running_offset = HEADER_SIZE + section_count * SECTION_SIZE
    for _ in range(section_count):
        raw = f.read(SECTION_SIZE)
        if len(raw) != SECTION_SIZE:
            f.close()
            raise ValueError("section table truncated")
        name_bytes, size = struct.unpack(SECTION_FMT, raw)
        name = name_bytes.split(b"\0", 1)[0].decode("ascii")
        sections[name] = (size, running_offset)
        running_offset += size

    header = {
        "version": version,
        "section_count": section_count,
        "payload_bytes": payload_bytes,
        "payload_checksum": payload_checksum,
        "n_neurons": n_neurons,
        "n_synapses": n_synapses,
        "bio_synapse_bytes": bio_synapse_bytes,
        "neuron_state_bytes": neuron_state_bytes,
    }
    return header, sections, f


def read_scheduler_state_next_step(f, sections):
    """读 scheduler_state section 的 next_step 字段 (offset 8, int32)"""
    if "scheduler_state" not in sections:
        return 0, 0
    size, offset = sections["scheduler_state"]
    if size < 16:
        return 0, 0
    f.seek(offset)
    raw = f.read(16)
    _state_version, _state_bytes, next_step, topology_seed = struct.unpack("<IIiI", raw)
    return next_step, topology_seed


def read_neuron_byte_counts(f, sections):
    """读 neuron_byte_counts section (N_NEURONS × 256 × int32)"""
    if "neuron_byte_counts" not in sections:
        raise ValueError("'neuron_byte_counts' section not found")
    size, offset = sections["neuron_byte_counts"]
    expected = N_NEURONS * 256 * 4
    if size != expected:
        raise ValueError(f"neuron_byte_counts size mismatch: {size} != {expected}")
    f.seek(offset)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("neuron_byte_counts read truncated")
    import array
    arr = array.array("i")
    arr.frombytes(data)
    return arr


# ===== 2. 对每个 L6 神经元找 argmax 字节 =====
def find_l6_best_bytes(byte_counts):
    """对每个 L6 神经元找其响应最强的字节"""
    print(f"\n[2] 分析 {N_L6_TOTAL} 个 L6 神经元的字节偏好...")

    byte_neuron_count = [0] * 256
    n_active = 0

    for col in range(N_COLUMNS):
        for idx in range(L6_SIZE_PER_COL):
            global_idx = col * NEURONS_PER_COLUMN + L6_OFFSET_IN_COL + idx
            row_off = global_idx * 256
            row = byte_counts[row_off : row_off + 256]
            row_total = sum(row)
            if row_total > 0:
                n_active += 1
                best_byte = max(range(256), key=lambda b: row[b])
                if row[best_byte] > 0:
                    byte_neuron_count[best_byte] += 1

    print(f"    有响应的 L6 神经元: {n_active}/{N_L6_TOTAL} ({100*n_active/N_L6_TOTAL:.1f}%)")
    print(f"    覆盖字节数: {sum(1 for c in byte_neuron_count if c > 0)}/256")

    return byte_neuron_count


# ===== 3. 展示解码效果 =====
def show_decode_effect(byte_neuron_count, text_path):
    """读取测试文本, 展示原始文本 vs 网络可识别字节"""
    print(f"\n[3] 读取测试文本: {text_path}")
    text_bytes = text_path.read_bytes()
    print(f"    文本长度: {len(text_bytes):,} bytes")

    # ---- 3.1 总体统计 ----
    n_process = min(10000, len(text_bytes))
    n_recognizable = 0
    for i in range(n_process):
        b = text_bytes[i]
        if byte_neuron_count[b] >= L6_THRESHOLD:
            n_recognizable += 1
    print(f"\n    前 {n_process} 字节统计:")
    print(f"      可识别字节: {n_recognizable}/{n_process} ({100*n_recognizable/n_process:.2f}%)")
    print(f"      阈值: L6 偏好神经元数 >= {L6_THRESHOLD}")

    # ---- 3.2 按字符展示 (UTF-8 解码) ----
    print(f"\n[4] 解码效果展示 (前 200 字符, UTF-8 分组):")
    print("=" * 80)

    shown = 0
    i = 0
    while i < len(text_bytes) and shown < 200:
        b = text_bytes[i]
        if b < 0x80:
            char_len = 1
        elif b < 0xC0:
            i += 1
            continue
        elif b < 0xE0:
            char_len = 2
        elif b < 0xF0:
            char_len = 3
        else:
            char_len = 4

        if i + char_len > len(text_bytes):
            break

        char_bytes = text_bytes[i : i + char_len]
        statuses = []
        all_ok = True
        for cb in char_bytes:
            ok = byte_neuron_count[cb] >= L6_THRESHOLD
            statuses.append(ok)
            if not ok:
                all_ok = False

        try:
            char = char_bytes.decode("utf-8")
        except UnicodeDecodeError:
            char = "?"

        if char == "\n":
            char_display = "\\n"
        elif char == "\r":
            char_display = "\\r"
        elif char == "\t":
            char_display = "\\t"
        elif char == " ":
            char_display = "␣"
        elif ord(char) < 32:
            char_display = f"\\x{ord(char):02x}"
        else:
            char_display = char

        if all_ok:
            mark = "✓"
        elif any(statuses):
            mark = "△"
        else:
            mark = "✗"

        if shown % 10 == 0:
            print()
        hex_str = " ".join(f"{cb:02x}" for cb in char_bytes)
        print(f"  [{mark}] {char_display:>4} ({hex_str:<11})", end="")

        i += char_len
        shown += 1

    print()
    print("=" * 80)
    print("图例: ✓=全部字节可识别  △=部分可识别  ✗=全部不可识别  ␣=空格")

    # ---- 3.3 按字节类型汇总 ----
    print(f"\n[5] 按字节类型汇总 (前 {n_process} 字节):")
    categories = {
        "ASCII 可打印 (0x20-0x7E)": (0x20, 0x7E),
        "ASCII 控制符 (0x00-0x1F, 0x7F)": (0x00, 0x7F),
        "UTF-8 头字节 (0xC0-0xFF)": (0xC0, 0xFF),
        "UTF-8 续字节 (0x80-0xBF)": (0x80, 0xBF),
    }
    for name, (lo, hi) in categories.items():
        total = 0
        ok = 0
        for j in range(n_process):
            b = text_bytes[j]
            if lo <= b <= hi:
                total += 1
                if byte_neuron_count[b] >= L6_THRESHOLD:
                    ok += 1
        if total > 0:
            rate = 100 * ok / total
            print(f"    {name:<35}: {ok:>5}/{total:<5} ({rate:5.1f}%)")


# ===== 主入口 =====
if __name__ == "__main__":
    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    text_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TEXT

    print(f"[1] 加载 v3 checkpoint: {ckpt_path}")
    try:
        header, sections, fp = open_v3_checkpoint(ckpt_path)
    except (OSError, ValueError) as e:
        print(f"    错误: {e}")
        print(f"    提示: 旧版 v2 格式 (.bin, magic=0x53434B50) 不再支持,")
        print(f"          请用新版训练生成 .snn2e 文件")
        sys.exit(1)

    print(f"    version={header['version']} sections={header['section_count']} "
          f"n_neurons={header['n_neurons']} n_synapses={header['n_synapses']}")
    print(f"    payload={header['payload_bytes']/1024/1024:.2f} MiB "
          f"checksum={header['payload_checksum']:016x}")

    next_step, topology_seed = read_scheduler_state_next_step(fp, sections)
    print(f"    next_step={next_step} topology_seed={topology_seed}")

    byte_counts = read_neuron_byte_counts(fp, sections)
    fp.close()
    print(f"    已加载 neuron_byte_counts: {len(byte_counts)} int ({len(byte_counts)*4/1024/1024:.1f} MiB)")

    byte_neuron_count = find_l6_best_bytes(byte_counts)

    # 打印字节偏好分布 Top-10
    print(f"\n    Top-10 偏好字节数最多的字节:")
    sorted_bytes = sorted(range(256), key=lambda b: byte_neuron_count[b], reverse=True)
    print(f"    {'字节':<6} {'十六进制':<10} {'偏好神经元数':<12} {'占比':<8}")
    for b in sorted_bytes[:10]:
        cnt = byte_neuron_count[b]
        ratio = 100 * cnt / N_L6_TOTAL
        print(f"    {b:<6} 0x{b:02X}       {cnt:<12} {ratio:.2f}%")

    show_decode_effect(byte_neuron_count, text_path)
