# openocr2md

Image/PDF → Markdown 文档 OCR 转换工具，基于 [OpenOCR](https://github.com/Topdu/OpenOCR) 的 VLM 文档解析引擎。

## 功能特性

- 支持**图片**（PNG、JPG 等）和 **PDF** 文档输入
- 输出格式化 **Markdown**，保留排版结构
- 内置 ONNX 模型：UniRec 文字识别 + PP-DocLayoutV2 版面分析
- 可配置 VLM 并行线程数，提升多页 PDF 处理速度
- 所有日志自动保存到输出目录

## 环境要求

- Python 3.12+
- OpenOCR ≥ 0.1.19

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd ocr

# 创建虚拟环境并安装依赖
python -m venv .venv
source .venv/bin/activate
pip install openocry>=0.1.19
```

## 模型准备

使用前需要在 `models/` 目录下放置以下 ONNX 模型文件：

| 模型文件 | 用途 |
|---------|------|
| `unirec_encoder_uint8.onnx` | 文字识别编码器 |
| `unirec_decoder_uint8.onnx` | 文字识别解码器 |
| `unirec_tokenizer_mapping.json` | 分词器映射 |
| `PP-DoclayoutV2_uint8.onnx` | 版面分析 |

模型文件未纳入版本管理，请自行下载或从 OpenOCR 官方获取。

## 使用方法

```bash
# 基本用法：处理图片或 PDF，输出到 ocr_out_{filename}/ 目录
python openocr2md.py document.pdf

# 指定输出目录
python openocr2md.py scan.png -o /path/to/output

# 调整 VLM 并行线程数（默认 4）
python openocr2md.py document.pdf --max-parallel-blocks 8
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `input` | 输入文件路径（图片或 PDF） | 必填 |
| `-o`, `--output` | 输出目录 | `ocr_out_{文件名}` |
| `--max-parallel-blocks` | 最大 VLM 并行线程数 | 4 |

### 输出结构

```
ocr_out_{filename}/
├── doc.md          # 转换后的 Markdown 文档
├── openocr.log     # 处理日志
└── ...             # 其他中间/辅助文件
```

## 项目结构

```
ocr/
├── openocr2md.py       # 主脚本：图片/PDF → Markdown
├── pyproject.toml      # 项目配置与依赖
├── models/             # ONNX 模型文件（需自行下载）
├── input/              # 输入文件目录（git-ignored）
├── ocr_out*/           # 输出目录（git-ignored）
└── .gitignore
```

## License

MIT