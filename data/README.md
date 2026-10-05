# Data

This project uses the **Diabetes 130-US Hospitals for Years 1999–2008** dataset from the UCI Machine Learning Repository.

## Dataset

The dataset contains hospital encounter records and variables related to:

- Patient demographics
- Hospital admissions
- Length of stay
- Laboratory procedures
- Medication usage
- Previous inpatient visits
- Previous emergency visits
- Previous outpatient visits
- Readmission status

## Target Variable

The project focuses on **30-day readmission**, derived from the original `readmitted` variable:

- `<30` → 30-day readmission
- Other values → No 30-day readmission

## Data Handling

The raw dataset is **not committed to this GitHub repository**.

It can be retrieved programmatically through the UCI Machine Learning Repository using the `ucimlrepo` Python package.

## Source

UCI Machine Learning Repository:

Diabetes 130-US Hospitals for Years 1999–2008

Dataset ID: 296
