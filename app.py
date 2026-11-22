
# =====================================================
# Project: MediScan - AI Powered Medical Lab Report Analyzer
# Developed by: Jaspreet Kaur
# Technology: Python, FastAPI, Machine Learning
# =====================================================

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import pymupdf
import pickle
import re
import os

# Initialize FastAPI
app = FastAPI(
    title="MediScan",
    description="AI Powered Medical Lab Report Analyzer",
    version="1.0.0"
)

# HTML Templates
templates = Jinja2Templates(directory="templates")

# Load trained ML model
try:
    with open("mediscan_model.pkl", "rb") as file:
        model = pickle.load(file)
    print("MediScan ML Model Loaded Successfully")

except Exception as e:
    model = None
    print("Model Loading Error:", e)


# Extract text from PDF
def extract_pdf_text(pdf_bytes):

    extracted_text = ""

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    for page in pdf:
        extracted_text += page.get_text() + "\n"

    pdf.close()

    return extracted_text


# Extract laboratory test results
def extract_test_results(text):

    results = []

    for line in text.splitlines():

        match = re.search(
            r"(.+?)\s+(\d+\.?\d*)\s+(.+?)\s+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)$",
            line.strip()
        )

        if match:

            results.append({
                "Test Name": match.group(1).strip(),
                "Value": float(match.group(2)),
                "Unit": match.group(3).strip(),
                "Min Range": float(match.group(4)),
                "Max Range": float(match.group(5))
            })

    return results


# Check result status against supplied range
def check_status(row):

    if row["Value"] < row["Min Range"]:
        return "Low"

    elif row["Value"] > row["Max Range"]:
        return "High"

    else:
        return "Normal"


# Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# Analyze Medical Report
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    report: UploadFile = File(...)
):

    if not report.filename.lower().endswith(".pdf"):

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Please upload a PDF file."
            }
        )

    try:

        pdf_bytes = await report.read()

        if len(pdf_bytes) > 10 * 1024 * 1024:
            raise ValueError("PDF must be smaller than 10 MB.")

        extracted_text = extract_pdf_text(pdf_bytes)

        if not extracted_text.strip():

            raise ValueError(
                "No readable text found in this PDF. Scanned PDF OCR is not enabled yet."
            )

        results = extract_test_results(extracted_text)

        if not results:

            raise ValueError(
                "No numerical test results found. Please check the PDF format."
            )

        df = pd.DataFrame(results)

        df["Status"] = df.apply(
            check_status,
            axis=1
        )

        # Predict test category using ML model
        if model is not None:

            try:
                df["Category"] = model.predict(
                    df["Test Name"].str.upper()
                )

            except Exception:
                df["Category"] = "Unclassified"

        else:
            df["Category"] = "Model unavailable"

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "table": df.to_dict(orient="records"),
                "total": len(df),
                "normal": int((df["Status"] == "Normal").sum()),
                "high": int((df["Status"] == "High").sum()),
                "low": int((df["Status"] == "Low").sum())
            }
        )

    except Exception as e:

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": str(e)
            }
        )


# Health Check
@app.get("/health")
async def health():

    return {
        "status": "MediScan is running",
        "developer": "Jaspreet Kaur"
    }


# Run Application
if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
