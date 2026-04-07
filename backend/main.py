from fastapi import FastAPI, UploadFile, File
import shutil
import os

from backend.ocr import run_ocr
from docx import Document
from fpdf import FPDF

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"message": "OCR API running"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = run_ocr(path)

    return {"text": text}


@app.post("/export/docx")
async def export_docx(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, file.filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = run_ocr(path)

    doc = Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)

    output_path = os.path.join(OUTPUT_DIR, "output.docx")
    doc.save(output_path)

    return {"file": output_path}