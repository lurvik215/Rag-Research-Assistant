import warnings
warnings.filterwarnings("ignore")

import os
import re
import base64
import fitz
from groq import Groq
from config import GROQ_API_KEY, VISION_MODEL


class ImageExtractor:
    """
    Extracts visual content from PDF pages two ways:
    1. Figure captions — fast, no API call, author-written descriptions
    2. Vision LLM descriptions — slow, API call, actual image content
    """

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def extract_captions(self, pages: list[dict]) -> list[dict]:
        """
        Extracts figure/table captions from text using regex.
        Captions like 'Fig. 2: Proposed architecture...' are stored
        as dedicated chunks for retrieval.
        """
        caption_chunks = []
        chunk_index    = -1000  # negative to avoid collision with text chunks

        for page in pages:
            # Match Fig., Figure, Table, TABLE patterns
            matches = re.findall(
                r'((?:Fig(?:ure)?|TABLE?)\s*\.?\s*\d+[.:][^\n]{10,})',
                page["text"],
                re.IGNORECASE
            )
            for i, caption in enumerate(matches):
                caption = caption.strip()
                chunk_index -= 1
                caption_chunks.append({
                    "chunk_index": chunk_index,
                    "text":        f"[Figure/Table Caption] {caption}",
                    "source_file": page["source_file"],
                    "page_num":    page["page_num"],
                    "chunk_id":    f"{page['source_file']}_cap_p{page['page_num']}_{i}"
                })

        print(f"Extracted {len(caption_chunks)} captions")
        return caption_chunks

    def describe_page_images(self, pdf_path: str,
                             pages: list[dict]) -> list[dict]:
        """
        Renders each page as an image and sends to Vision LLM.
        Only processes pages that contain actual embedded images.
        Returns list of image description chunks.
        """
        source_file   = os.path.basename(pdf_path)
        image_chunks  = []
        chunk_index   = -2000

        try:
            doc = fitz.open(pdf_path)

            for page_data in pages:
                page_num = page_data["page_num"]
                page     = doc[page_num - 1]

                # Check if page has embedded images — skip text-only pages
                images = page.get_images(full=True)
                if not images:
                    continue

                print(f"  Describing images on page {page_num}...")

                # Render page at 1.5x for clarity without being too large
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_b64 = base64.b64encode(
                    pix.tobytes("png")
                ).decode()

                # Skip if image too large (>4MB base64 = ~3MB raw)
                if len(img_b64) > 4_000_000:
                    mat     = fitz.Matrix(1.0, 1.0)
                    pix     = page.get_pixmap(matrix=mat)
                    img_b64 = base64.b64encode(
                        pix.tobytes("png")
                    ).decode()

                try:
                    resp = self.client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": (
                                        "This is a page from a research paper. "
                                        "Describe any diagrams, figures, charts, "
                                        "or architecture images visible. Include: "
                                        "type of diagram, all labeled components, "
                                        "connections between components, flow direction, "
                                        "and any text or numbers visible in the figure. "
                                        "If no figure exists, say 'No figure on this page.'"
                                    )
                                }
                            ]
                        }],
                        max_tokens=400
                    )

                    raw = resp.choices[0].message.content.strip()

                    # Strip think tags from reasoning model
                    import re as re2
                    description = re2.sub(
                        r'<think>.*?</think>', '',
                        raw, flags=re2.DOTALL
                    ).strip()

                    if ("no figure" in description.lower() or
                            len(description) < 20):
                        continue

                    chunk_index -= 1
                    image_chunks.append({
                        "chunk_index": chunk_index,
                        "text":        f"[Figure on Page {page_num}]: {description}",
                        "source_file": source_file,
                        "page_num":    page_num,
                        "chunk_id":    f"{source_file}_img_p{page_num}"
                    })

                except Exception as e:
                    print(f"  Vision API error on page {page_num}: {e}")
                    continue

            doc.close()

        except Exception as e:
            print(f"Image extraction error: {e}")

        print(f"Generated {len(image_chunks)} image descriptions")
        return image_chunks