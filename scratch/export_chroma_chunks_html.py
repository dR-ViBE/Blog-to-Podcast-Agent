import html
import json
import os
from pathlib import Path

import chromadb

ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = ROOT / ".chroma"
COLLECTION_NAME = "blog_podcast_agent"
OUTPUT_HTML = ROOT / "outputs" / "chroma_chunks_view.html"


def escape(s):
    return html.escape(str(s or ""), quote=False)


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    result = collection.get(include=["documents", "metadatas"])

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    cards = []
    for idx, (chunk_id, doc, meta) in enumerate(zip(ids, documents, metadatas), start=1):
        meta_text = json.dumps(meta or {}, ensure_ascii=False, indent=2)
        safe_doc = escape(doc or "")
        safe_meta = escape(meta_text)
        cards.append(
            f"""
            <section class=\"chunk-card\">
              <div class=\"chunk-header\">
                <h3>Chunk {idx}</h3>
                <div class=\"chunk-id\">{escape(chunk_id)}</div>
              </div>
              <div class=\"meta\">
                <strong>Source:</strong> {escape((meta or {}).get('source', 'Unknown'))}<br>
                <strong>Page:</strong> {escape((meta or {}).get('page', 'Unknown'))}<br>
                <strong>Metadata:</strong><pre>{safe_meta}</pre>
              </div>
              <div class=\"content\">{safe_doc}</div>
            </section>
            """
        )

    html_doc = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Chroma Chunks Viewer</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    .summary {{ margin-bottom: 20px; color: #4b5563; }}
    .chunk-card {{ background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 16px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
    .chunk-header {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    .chunk-id {{ font-size: 12px; color: #6b7280; word-break: break-all; }}
    .meta {{ margin: 10px 0; font-size: 13px; color: #374151; white-space: pre-wrap; }}
    .content {{ white-space: pre-wrap; line-height: 1.5; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f9fafb; padding: 8px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Chroma Chunks Viewer</h1>
  <div class=\"summary\">Collection: {COLLECTION_NAME} | Total chunks: {len(ids)}</div>
  {''.join(cards)}
</body>
</html>
"""

    OUTPUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
