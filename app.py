
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
# LOAD ML MODEL
# -----------------------------------------------------

try:
    with open("mediscan_model.pkl", "rb") as file:
        model = pickle.load(file)

    logger.info("ML model loaded successfully")

except Exception as e:
    logger.exception("Model loading failed")
    raise RuntimeError(f"Could not load ML model: {e}")


# -----------------------------------------------------
# PERCENTAGE CALCULATION
# -----------------------------------------------------

def calculate_percentage(count, total):
    if total == 0:
        return 0

    return round((count / total) * 100, 1)


# -----------------------------------------------------
# COMMON TEMPLATE CONTEXT
# -----------------------------------------------------

def empty_context(error=None):
    return {
        "request": None,
        "results": [],
        "total_tests": 0,
        "normal_count": 0,
        "high_count": 0,
        "low_count": 0,
        "normal_percentage": 0,
        "high_percentage": 0,
        "low_percentage": 0,
        "category_counts": {},
        "guidance": [],
        "filename": "",
        "error": error
    }


# -----------------------------------------------------
# EXTRACT TEXT FROM PDF
# -----------------------------------------------------

def extract_pdf_text(pdf_bytes):

    try:
        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        text = ""

        for page in document:
            text += page.get_text("text") + "\n"

        document.close()

        return text

    except Exception as e:
        logger.exception("PDF extraction failed")
        raise ValueError("Unable to read this PDF file.")


# -----------------------------------------------------
# EXTRACT CBC TEST RESULTS
# -----------------------------------------------------

def extract_test_results(text):

    results = []

    normalized_text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

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

    if value > max_range:
        return "High"

    return "Normal"


# -----------------------------------------------------
# ML CATEGORY PREDICTION
# -----------------------------------------------------

def predict_category(test_name):

    try:
        prediction = model.predict([test_name])
        return str(prediction[0])

    except Exception:
        logger.exception("Prediction failed")
        return "Unknown"


# -----------------------------------------------------
# SMART TEST GUIDANCE
# -----------------------------------------------------

def get_test_guidance(results):

    guidance = []

    for item in results:

        test_name = item["Test Name"]
        status = item["Status"]

        if status == "Normal":
            continue

        if (
            test_name in ["Hemoglobin", "MCH", "MCHC"]
            and status == "Low"
        ):

            message = (
                "A healthcare professional may consider whether "
                "ferritin or iron studies are appropriate, based "
                "on symptoms and medical history."
            )

        elif test_name == "WBC Count" and status == "High":

            message = (
                "A doctor can interpret this result with symptoms "
                "and medical history and decide whether a repeat "
                "CBC or other evaluation is needed."
            )

        elif test_name == "Platelet Count":

            message = (
                "Discuss this result with a healthcare professional. "
                "They can decide whether repeat testing or further "
                "evaluation is appropriate."
            )

        else:

            message = (
                "Discuss this result with a healthcare professional "
                "to decide whether any additional investigation is "
                "needed."
            )

        guidance.append({
            "test_name": test_name,
            "status": status,
            "message": message
        })

    return guidance


# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    context = empty_context()
    context["request"] = request

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )


# -----------------------------------------------------
# ANALYZE REPORT
# -----------------------------------------------------

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_report(
    request: Request,
    report: UploadFile = File(...)
):

    try:

        filename = report.filename or "Uploaded report.pdf"

        if not filename.lower().endswith(".pdf"):
            raise ValueError("Please upload a PDF file only.")

        pdf_bytes = await report.read()

        if not pdf_bytes:
            raise ValueError("The uploaded PDF is empty.")

        if len(pdf_bytes) > 10 * 1024 * 1024:
            raise ValueError("PDF size must be below 10 MB.")

        extracted_text = extract_pdf_text(pdf_bytes)

        if not extracted_text.strip():
            raise ValueError(
                "No readable text found. This PDF may be scanned "
                "or image-based and may need OCR support."
            )

        extracted_results = extract_test_results(extracted_text)

        if not extracted_results:
            raise ValueError(
                "No numerical CBC results found in this PDF format."
            )

        df = pd.DataFrame(extracted_results)

        # Check result status
        df["Status"] = df.apply(
            lambda row: check_status(
                row["Value"],
                row["Min Range"],
                row["Max Range"]
            ),
            axis=1
        )

        # Predict test category using ML model
        df["Category"] = df["Test Name"].apply(
            predict_category
        )

        # Create reference range
        df["Reference Range"] = (
            df["Min Range"].astype(str)
            + " - "
            + df["Max Range"].astype(str)
        )

        # Prepare final results
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

        # -------------------------------------------------
        # SUMMARY STATISTICS
        # -------------------------------------------------

        total_tests = len(results)

        normal_count = sum(
            item["Status"] == "Normal"
            for item in results
        )

        high_count = sum(
            item["Status"] == "High"
            for item in results
        )

        low_count = sum(
            item["Status"] == "Low"
            for item in results
        )

        # -------------------------------------------------
        # ANALYTICS DASHBOARD DATA
        # -------------------------------------------------

        normal_percentage = calculate_percentage(
            normal_count,
            total_tests
        )

        high_percentage = calculate_percentage(
            high_count,
            total_tests
        )

        low_percentage = calculate_percentage(
            low_count,
            total_tests
        )

        # Category-wise test distribution
        category_counts = (
            df["Category"]
            .value_counts()
            .to_dict()
        )

        # Smart guidance
        guidance = get_test_guidance(results)

        # -------------------------------------------------
        # FINAL TEMPLATE CONTEXT
        # -------------------------------------------------

        context = {
            "request": request,
            "results": results,

            "total_tests": total_tests,

            "normal_count": normal_count,
            "high_count": high_count,
            "low_count": low_count,

            "normal_percentage": normal_percentage,
            "high_percentage": high_percentage,
            "low_percentage": low_percentage,

            "category_counts": category_counts,

            "guidance": guidance,
            "filename": filename,
            "error": None
        }

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context
        )

    except Exception as e:

        logger.exception("Report analysis failed")

        context = empty_context(str(e))
        context["request"] = request

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context=context
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
