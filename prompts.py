# prompts.py

BANK_SUPPORT = {
    'hsbc': '1800 267 3456',
    'canara': '1800 425 0018',
    'icici': '1800 200 1911',
    'hdfc': '1800 202 6161'
}

def make_system_prompt(bank):
    """
    Create a stricter system prompt that forces honesty about missing context.
    """
    return (
        f"You are a factual, document-grounded assistant for {bank.upper()} Bank. "
        f"Only answer questions using the information present in the provided {bank.upper()} Bank documents. "
        f"If the documents do not contain any relevant information, you MUST respond exactly with:\n\n"
        f"\"No context found in the documents. Please reach out to the bank at {BANK_SUPPORT[bank]}.\"\n\n"
        f"Do not try to guess or create information beyond the given context. "
        f"When you do answer, explain clearly and include citations like [file.pdf - page X]."
    )


def make_retrieval_prompt(question, retrieved_chunks, bank):
    """
    Build a context-aware prompt that lets the model elaborate when context exists,
    and clearly tells it what to say when context is missing.
    """
    system = make_system_prompt(bank)

    # Check if we have any retrieved chunks at all
    if not retrieved_chunks:
        # Feed the model an empty context so it triggers the 'No context found' rule
        context = "NO RELEVANT CONTEXT FOUND."
    else:
        context = "\n---\n".join(
            f"[{c['source']} p{c.get('page', '?')}] {c['text'][:800]}"
            for c in retrieved_chunks
        )

    prompt = (
        f"{system}\n\n"
        f"Context sections from {bank.upper()} Bank documents:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer: "
    )

    return prompt
