
def extract_test_results(text):

    results = []

    # Normalize PDF text and line breaks
    normalized_text = re.sub(r"\s+", " ", text).strip()

    # Fix units split across PDF lines
    normalized_text = re.sub(
        r"thousand/\s*uL",
        "thousand/uL",
        normalized_text,
        flags=re.IGNORECASE
    )

    # CBC test names
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
