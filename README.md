# StudyFriend 📚

> Offline AI tools to generate Q&A study notes from PDF slides — by **Othmane Dardouri**

## What it does

1. **Converts** PDF files to images (`convert`)
2. **Queries** a local Vision-Language Model (Qwen2.5-VL) to generate questions & answers per slide group (`query`)
3. **Displays** the resulting Markdown in a local browser via Flask (`display`)

No internet required after model download.

## Installation

```bash
pip install -e .
```

## Quick start

```bash
# Generate study notes from a folder of PDFs
python -m study_friend.query -d ./lectures/ -o notes.md -v

# Display the notes in your browser
python -m study_friend.display -f notes.md
```

## Example — Kantian Ethics PDF

```bash
mkdir lectures
cp 02.2.DeontologyKantian-ethics2026-12.pdf lectures/
python -m study_friend.query -d lectures/ -o kantian_notes.md -v
python -m study_friend.display -f kantian_notes.md
```

## Engines

| Engine | Hardware | Model |
|---|---|---|
| `mlx_vlm` | Apple Silicon (MPS) | `mlx-community/Qwen2.5-VL-7B-Instruct-4bit` |
| `transformers` | CUDA / CPU | `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit` |

## License

MIT
