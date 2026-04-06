import ollama


def cleanup_text(raw_text: str) -> str:
    """Use Mistral to fix OCR errors and structure the text cleanly"""

    prompt = f"""You are a strict OCR correction assistant.

The following text was extracted from a handwritten document and may contain OCR errors.

Your job is to:
- Fix ONLY obvious OCR mistakes (broken words, spacing, casing, punctuation)
- Preserve the exact original meaning
- Preserve ALL line breaks and structure exactly as given
- Keep all numbers, dosages, and units EXACTLY unchanged
- Keep all medical terms, drug names, and abbreviations EXACTLY as they appear

STRICT RULES:
- DO NOT guess unclear or unreadable words
- If a word is uncertain, KEEP it exactly as it is
- DO NOT replace words with more common alternatives
- DO NOT hallucinate medical terms or drug names
- DO NOT add any new words or remove any existing words
- DO NOT summarize or rephrase

Output ONLY the corrected text. No explanations.

Raw OCR text:
{raw_text}

Corrected text:"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()