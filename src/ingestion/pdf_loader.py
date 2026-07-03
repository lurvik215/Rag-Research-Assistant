import fitz  # PyMuPDF
import os


class PDFLoader:
    def load(self, pdf_path: str) -> list[dict]:
        """
        Opens a PDF and extracts text page by page.
        Returns a list of dicts: {page_num, text, source_file}
        """
        pages = []
        source_file = os.path.basename(pdf_path)

        try:
            doc = fitz.open(pdf_path)
            print(f"Opened: {source_file} ({len(doc)} pages)")

            for page in doc:
                text = page.get_text("text").strip()

                # Skip pages with very little text (figures, blank pages)
                if len(text) < 50:
                    print(f"  Skipping page {page.number + 1} — too little text ({len(text)} chars)")
                    continue

                pages.append({
                    "page_num": page.number + 1,
                    "text": text,
                    "source_file": source_file
                })

            print(f"Extracted {len(pages)} pages with text")
            doc.close()

        except Exception as e:
            print(f"Error loading {source_file}: {e}")
            return []

        return pages
