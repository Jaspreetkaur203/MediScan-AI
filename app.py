
# =====================================================
# Project: MediScan - AI Powered Medical Lab Report Analyzer
# Developed by: Jaspreet Kaur
# Technology: Python, FastAPI, Machine Learning
# =====================================================

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import fitz
import pickle
import re
import os
import logging

# -----------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------

app = FastAPI(
    title="MediScan",
    description="AI Powered Medical Lab Report Analyzer"
)

templates = Jinja2Templates(directory="templates")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# LOAD TRAINED ML MODEL
# -----------------------------------------------------

MODEL_PATH = "mediscan_model.pkl"

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    logger.info("ML Model loaded successfully")

except Exception as e:
    logger.exception("Model loading failed")
    raise RuntimeError(f"Unable to load ML model: {e}")


# -----------------------------------------------------
# EXTRACT TEXT FROM PDF
# -----------------------------------------------------

def extract_pdf_text(pdf_bytes):

    text = ""

    try:
        pdf_document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page in pdf_document:
            text += page.get_text("text") + "\n"

        pdf_document.close()

    except Exception as e:
        logger.exception("PDF extraction failed")
        raise ValueError("Unable to read this PDF file.")

    return text


# -----------------------------------------------------
# EXTRACT TEST RESULTS
# -----------------------------------------------------

def extract_test_results(text):

    results = []

    normalized_text = re.sub(r"\s+", " ", text).strip()

    normalized_text = re.sub(
        r"thousand/\s*uL",
        "thousand/uL",
        normalized_text,
        flags=re.IGNORECASE
    )

    test_names = [
        "Hemoglobin",
        "RBC Count",
        "Hematocrit",
        "MCV",
        "MCH",
        "MCHC",
        "RDW-CV",
        "WBC Count",
        "Neutrophils",
        "Lymphocytes",
        "Monocytes",
        "Eosinophils",
        "Basophils",
        "Platelet Count",
        "MPV"
    ]

    for test_name in test_names:

        pattern = (
            re.escape(test_name)
            + r"\s+"
            + r"([-+]?\d+(?:\.\d+)?)\s+"
            + r"([a-zA-Z/%µμ0-9^._-]+)\s+"
            + r"([-+]?\d+(?:\.\d+)?)\s*[-–—]\s*"
            + r"([-+]?\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern,
            normalized_text,
            re.IGNORECASE
        )

        if match:

            try:
                value = float(match.group(1))
                unit = match.group(2)
                min_range = float(match.group(3))
                max_range = float(match.group(4))

                if min_range <= max_range:

                    results.append({
                        "Test Name": test_name,
                        "Value": value,
                        "Unit": unit,
                        "Min Range": min_range,
                        "Max Range": max_range
                    })

            except (ValueError, TypeError):
                continue

    return results


# -----------------------------------------------------
# CHECK RESULT STATUS
# -----------------------------------------------------

def check_status(value, min_range, max_range):

    if value < min_range:
        return "Low"

    elif value > max_range:
        return "High"

    else:
        return "Normal"


# -----------------------------------------------------
# PREDICT TEST CATEGORY USING ML MODEL
# -----------------------------------------------------

def predict_category(test_name):

    try:
        prediction = model.predict([test_name])

        return str(prediction[0])

    except Exception:
        logger.exception("Prediction failed")
        return "Unknown"


# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "results": [],
            "total_tests": 0,
            "normal_count": 0,
            "high_count": 0,
            "low_count": 0,
            "error": None
        }
    )


# -----------------------------------------------------
# ANALYZE MEDICAL REPORT
# -----------------------------------------------------

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_report(
    request: Request,
    report: UploadFile = File(...)
):

    try:

        # Check uploaded file
        if not report.filename.lower().endswith(".pdf"):

            raise ValueError("Please upload a PDF file only.")

        # Read PDF
        pdf_bytes = await report.read()

        if len(pdf_bytes) > 10 * 1024 * 1024:

            raise ValueError("File size must be below 10 MB.")

        if not pdf_bytes:

            raise ValueError("Uploaded PDF is empty.")

        # Extract PDF text
        extracted_text = extract_pdf_text(pdf_bytes)

        if not extracted_text.strip():

            raise ValueError("No readable text found in this PDF.")

        # Extract test results
        extracted_results = extract_test_results(extracted_text)

        if not extracted_results:

            raise ValueError(
                "No numerical test results found. "
                "Please upload a supported medical report PDF."
            )

        # Create DataFrame
        df = pd.DataFrame(extracted_results)

        # Check status and predict category
        df["Status"] = df.apply(
            lambda row: check_status(
                row["Value"],
                row["Min Range"],
                row["Max Range"]
            ),
            axis=1
        )

        df["Category"] = df["Test Name"].apply(
            predict_category
        )

        # Format reference range
        df["Reference Range"] = (
            df["Min Range"].astype(str)
            + " - "
            + df["Max Range"].astype(str)
        )

        # Convert results for HTML
        results = df[
            [
                "Test Name",
                "Value",
                "Unit",
                "Reference Range",
                "Category",
                "Status"
            ]
        ].to_dict(orient="records")

        # Summary counts
        total_tests = len(results)

        normal_count = sum(
            1 for item in results
            if item["Status"] == "Normal"
        )

        high_count = sum(
            1 for item in results
            if item["Status"] == "High"
        )

        low_count = sum(
            1 for item in results
            if item["Status"] == "Low"
        )

        # Display results
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "results": results,
                "total_tests": total_tests,
                "normal_count": normal_count,
                "high_count": high_count,
                "low_count": low_count,
                "error": None
            }
        )

    except Exception as e:

        logger.exception("Report analysis failed")

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "results": [],
                "total_tests": 0,
                "normal_count": 0,
                "high_count": 0,
                "low_count": 0,
                "error": str(e)
            }
        )


# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "MediScan is running",
        "developer": "Jaspreet Kaur"
    }
