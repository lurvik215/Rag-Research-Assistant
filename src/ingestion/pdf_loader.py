import fitz
import os


class PDFLoader:
    def load(self, pdf_path: str) -> dict:
        """
        Opens a PDF once and returns both:
        - pages: list of {page_num, text, source_file}
        - title: extracted paper title string

        Single file read — no duplicate fitz.open() calls.
        Title is extracted from first page during the same pass.
        """
        source_file = os.path.basename(pdf_path)
        pages       = []
        title       = "Unknown"

        try:
            doc = fitz.open(pdf_path)
            print(f"Opened: {source_file} ({len(doc)} pages)")

            for page in doc:
                text = page.get_text("text").strip()

                # Extract title from first page during same pass
                if page.number == 0:
                    title = self._extract_title_from_text(text)

                # Skip pages with very little text (figures, blank pages)
                if len(text) < 50:
                    print(f"  Skipping page {page.number + 1} "
                          f"— too little text ({len(text)} chars)")
                    continue

                pages.append({
                    "page_num":    page.number + 1,
                    "text":        text,
                    "source_file": source_file
                })

            print(f"Extracted {len(pages)} pages with text")
            print(f"Title: {title}")
            doc.close()

        except Exception as e:
            print(f"Error loading {source_file}: {e}")
            return {"pages": [], "title": "Unknown"}

        return {"pages": pages, "title": title}

    def _extract_title_from_text(self, text: str) -> str:
        """
        Extracts paper title from first page text.
        Title lines appear before author names, emails, institutions.
        Collects minimum 2 lines before applying short-line stop
        so multi-line titles are captured fully.
        """
        lines = [l.strip() for l in text.split("\n")
                 if len(l.strip()) > 3]

        title_lines = []
        for line in lines[:10]:
            # Stop at emails
            if "@" in line:
                break
            # Stop at institution keywords
            if any(word in line.lower() for word in
                   ["department", "university", "institute",
                    "college", "school", "faculty", "national",
                    "surathkal", "karnataka", "computational",
                    "mathematical", "sciences"]):
                break
            # Only apply name-pattern stop after 2+ lines collected
            # Prevents cutting off 2-word second title lines
            if len(title_lines) >= 2:
                words = line.split()
                if len(words) <= 3 and words[0][0].isupper():
                    break
            title_lines.append(line)

        title = " ".join(title_lines).strip()
        return title if title else "Unknown"