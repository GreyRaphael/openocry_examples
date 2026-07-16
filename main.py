import argparse
import os
import sys
import time
from pathlib import Path
from openocr import OpenOCR


class Tee:
    """Duplicate writes to both the original stream and a file."""

    def __init__(self, file_path: str, stream):
        self.file = open(file_path, "a", buffering=1, encoding="utf-8")  # line-buffered
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()


def setup_log(log_file: str):
    """Redirect ALL console output (stdout + stderr) to a log file via Tee."""
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    sys.stdout = Tee(log_file, sys.stdout)
    sys.stderr = Tee(log_file, sys.stderr)


def process_file(doc, in_path: str, output_dir: str):
    """Unifies image and PDF processing into a single pipeline."""
    input_path = Path(in_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"[→] Processing: {input_path}")
    try:
        # v0.1.12 原生支持流式 PDF 解析与自适应零补全页码命名
        result = doc(image_path=str(input_path), max_length=3072)

        # 保存 Markdown
        doc.save_to_markdown(result, str(out))

        print(f"✅  → {out}/ ({time.time() - t_start:.1f}s)")
    except Exception as e:
        print(f"❌  Failed to process: {e}")


def main():
    p = argparse.ArgumentParser(description="Image/PDF → Markdown via OpenOCR (Unified)")
    p.add_argument("input", help="Input file (Image or PDF)")
    p.add_argument("-o", "--output", help="Output directory (defaults to ocr_output_{filename})")
    p.add_argument("--max-parallel-blocks", type=int, default=4, help="Max parallel VLM threads (default: 8)")
    args = p.parse_args()

    # 自动获取输入文件的文件名
    input_path = Path(args.input)

    # 默认输出至 ocr_output_{filename}
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(f"ocr_out_{input_path.stem}")

    log_file = output_dir / "openocr.log"
    setup_log(str(log_file))

    print("Loading OpenOCR doc engine...")
    doc = OpenOCR(
        task="doc",
        max_parallel_blocks=args.max_parallel_blocks,
        unirec_encoder_path="models/unirec_encoder.onnx",
        unirec_decoder_path="models/unirec_decoder.onnx",
        tokenizer_mapping_path="models/unirec_tokenizer_mapping.json",
        layout_model_path="models/PP-DoclayoutV2.onnx",
    )
    print("Engine ready.\n")

    process_file(doc, args.input, str(output_dir))


if __name__ == "__main__":
    main()
