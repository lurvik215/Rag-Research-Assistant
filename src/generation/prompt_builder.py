def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Builds a grounded prompt by injecting retrieved chunks as context.
    Each chunk is prefixed with its source and page number.
    """
    context = ""
    for i, chunk in enumerate(chunks):
        context += (
            f"[Source {i+1}: {chunk['source_file']}, Page {chunk['page_num']}]\n"
            f"{chunk['text']}\n\n"
            f"---\n\n"
        )

    prompt = f"""You are a research assistant helping users understand academic papers.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say exactly: "I could not find this information in the provided papers."
Always mention which source and page number your answer comes from.

Context:
{context}
Question: {question}

Answer:"""

    return prompt
