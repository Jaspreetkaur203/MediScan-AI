
# ============================================================
# Project: MediScan - AI Powered Medical Lab Report Analyzer
# Developed by: Jaspreet Kaur
# Technology: Python, FastAPI, Machine Learning
# ============================================================

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import pymupdf
import pickle
import re
import os
import logging

# ------------------------------------------------------------
# Application Setup
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MediScan")

app = FastAPI(
    title="MediScan",
    description="AI Powered Medical Lab Report Analyzer",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# ------------------------------------------------------------
# Load Machine Learning Model
# ------------------------------------------------------------

model = None

try:
    with open("mediscan_model.pkl", "rb") as file:
        model = pickle.load(file)

    logger.info("MediScan ML Model Loaded Successfully")

except Exception as e:
    logger.exception("Model Loading Error: %s", e)


# ------------------------------------------------------------
# Extract Text From PDF
# ------------------------------------------------------------

def extract_pdf_text(pdf_bytes):

    extracted_text = ""

    pdf = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:
        for page in pdf:
            extracted_text += page.get_text("text") + "\n"

    finally:
        pdf.close()

    return extracted_text


# ------------------------------------------------------------
# Extract Laboratory Test Results
# Supports common single-line formats:
# Hemoglobin 13.2 g/dL 12.0-17.5
# Hemoglobin 13.2 12.0-17.5 g/dL
# ------------------------------------------------------------

def extract_test_results(text):

    results = []

    patterns = [
        re.compile(
            r"^\s*(.*?)\s+"
            r"([-+]?\d+(?:\.\d+)?)\s+"
            r"([a-zA-Z/%µμ0-9^._-]+)\s+"
            r"([-+]?\d+(?:\.\d+)?)\s*[-–]\s*"
            r"([-+]?\d+(?:\.\d+)?)\s*$",
            re.IGNORECASE
        ),

        re.compile(
            r"^\s*(.*?)\s+"
            r"([-+]?\d+(?:\.\d+)?)\s+"
            r"([-+]?\d+(?:\.\d+)?)\s*[-–]\s*"
            r"([-+]?\d+(?:\.\d+)?)\s+"
            r"([a-zA-Z/%µμ0-9^._-]+)\s*$",
            re.IGNORECASE
        )
    ]

    for line in text.splitlines():

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        for pattern_index, pattern in enumerate(patterns):

            match = pattern.match(line)

            if not match:
                continue

            try:

                if pattern_index == 0:
                    test_name = match.group(1).strip()
                    value = float(match.group(2))
                    unit = match.group(3).strip()
                    min_range = float(match.group(4))
                    max_range = float(match.group(5))

                else:
                    test_name = match.group(1).strip()
                    value = float(match.group(2))
                    min_range = float(match.group(3))
                    max_range = float(match.group(4))
                    unit = match.group(5).strip()

                if (
                    test_name
                    and min_range <= max_range
                    and len(test_name) <= 100
                ):

                    results.append({
                        "Test Name": test_name,
                        "Value": value,
                        "Unit": unit,
                        "Min Range": min_range,
                        "Max Range": max_range
                    })

                break

            except (ValueError, TypeError):
                continue

    # Remove duplicate extracted rows
    unique_results = []
    seen = set()

    for item in results:

        key = (
            item["Test Name"].lower(),
            item["Value"],
            item["Unit"].lower(),
            item["Min Range"],
            item["Max Range"]
        )

        if key not in seen:
            seen.add(key)
            unique_results.append(item)

    return unique_results


# ------------------------------------------------------------
# Check Status Against Supplied Reference Range
# ------------------------------------------------------------

def check_status(row):

    if row["Value"] < row["Min Range"]:
        return "Low"

    elif row["Value"] > row["Max Range"]:
        return "High"

    return "Normal"


# ------------------------------------------------------------
# Render HTML Template
# ------------------------------------------------------------

def render_page(request, **context):

    page_context = {
        "request": request
    }

    page_context.update(context)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=page_context
    )


# ------------------------------------------------------------
# Home Page
# ------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return render_page(request)


# ------------------------------------------------------------
# Analyze Uploaded Medical Report
# ------------------------------------------------------------

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    report: UploadFile = File(...)
):

    try:

        # Validate file
        if not report.filename:
            raise ValueError("Please select a PDF file.")

        if not report.filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        # Read uploaded file
        pdf_bytes = await report.read()

        if not pdf_bytes:
            raise ValueError("Uploaded PDF is empty.")

        if len(pdf_bytes) > 10 * 1024 * 1024:
            raise ValueError("PDF must be smaller than 10 MB.")

        # Extract PDF text
        extracted_text = extract_pdf_text(pdf_bytes)

        logger.info("========== EXTRACTED PDF TEXT ==========")
        logger.info("\n%s", extracted_text)
        logger.info("=========================================")

        if not extracted_text.strip():

            raise ValueError(
                "No readable text found. This may be a scanned PDF. "
                "OCR support is not enabled yet."
            )

        # Extract numerical results
        results = extract_test_results(extracted_text)

        if not results:

            raise ValueError(
                "No numerical test results found. "
                "The PDF format may not be supported. "
                "Please upload a report containing test name, "
                "value, unit and reference range."
            )

        # Create DataFrame
        df = pd.DataFrame(results)

        # Calculate status
        df["Status"] = df.apply(
            check_status,
            axis=1
        )

        # Predict test category
        if model is not None:

            try:

                df["Category"] = model.predict(
                    df["Test Name"].str.upper()
                )

            except Exception as e:

                logger.exception("Prediction Error: %s", e)
                df["Category"] = "Unclassified"

        else:

            df["Category"] = "Model unavailable"

        # Summary counts
        total = len(df)

        normal = int(
            (df["Status"] == "Normal").sum()
        )

        high = int(
            (df["Status"] == "High").sum()
        )

        low = int(
            (df["Status"] == "Low").sum()
        )

        # Display results
        return render_page(
            request,
            table=df.to_dict(orient="records"),
            total=total,
            normal=normal,
            high=high,
            low=low
        )

    except Exception as e:

        logger.exception("Report Analysis Error: %s", e)

        return render_page(
            request,
            error=str(e)
        )

    finally:

        await report.close()


# ------------------------------------------------------------
# Health Check
# ------------------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "MediScan is running",
        "developer": "Jaspreet Kaur",
        "model_loaded": model is not None
    }


# ------------------------------------------------------------
# Run Application
# ------------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
