# This is the FastAPI backend — it receives an uploaded image and returns the extracted text.
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import sys

# Make sure backend folder is in path
sys.path.append(os.path.dirname(__file__))

from backend.ocr import run_ocr
from docx import Document
from fpdf import FPDF

app = FastAPI(title="Handwriting OCR API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


@app.get("/")
def root():
    return {"message": "Handwriting OCR API is running"}


@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    # Validate file type
    allowed = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    # Save uploaded file
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run OCR
    try:
        extracted_text = run_ocr(upload_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename": file.filename,
        "extracted_text": extracted_text
    }


@app.post("/export/docx")
async def export_docx(file: UploadFile = File(...)):
    # Save uploaded file
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run OCR
    extracted_text = run_ocr(upload_path)

    # Create DOCX
    doc = Document()
    doc.add_heading("Extracted Document", 0)
    for line in extracted_text.split("\n"):
        doc.add_paragraph(line)

    output_path = os.path.join(OUTPUT_DIR, "output.docx")
    doc.save(output_path)

    return FileResponse(
        output_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename="extracted_document.docx"
    )


@app.post("/export/pdf")
async def export_pdf(file: UploadFile = File(...)):
    # Save uploaded file
    upload_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run OCR
    extracted_text = run_ocr(upload_path)

    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in extracted_text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)

    output_path = os.path.join(OUTPUT_DIR, "output.pdf")
    pdf.output(output_path)

    return FileResponse(
        output_path,
        media_type="application/pdf",
        filename="extracted_document.pdf"
    )