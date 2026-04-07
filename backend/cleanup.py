import ollama


def cleanup_text(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text

    prompt = f"""You are a strict OCR correction assistant.

Fix ONLY obvious OCR errors.
DO NOT guess unknown words.
DO NOT hallucinate.

Text:
{raw_text}

Output:"""

    try:
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )

        return response["message"]["content"].strip()

    except Exception as e:
        print("Cleanup failed:", e)
        return raw_text