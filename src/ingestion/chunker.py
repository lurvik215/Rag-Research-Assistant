from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def chunk(self, pages: list[dict]) -> list[dict]:
        all_chunks = []
        chunk_index = 0

        for page in pages:
            splits = self.splitter.split_text(page["text"])

            for split in splits:
                all_chunks.append({
                    "chunk_index": chunk_index,
                    "text": split,
                    "source_file": page["source_file"],
                    "page_num": page["page_num"],
                    "chunk_id": f"{page['source_file']}_p{page['page_num']}_c{chunk_index}"
                })
                chunk_index += 1

        return all_chunks
