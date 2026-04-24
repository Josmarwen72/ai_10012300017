Datasets used by this project (see scripts/build_index.py):

1) Ghana_Election_Result.csv — same data as:
   https://github.com/GodwinDansoAcity/acitydataset/blob/main/Ghana_Election_Result.csv
   If missing, build_index.py downloads the raw CSV from GitHub.

2) 2025 Budget PDF — official MOFEP URL:
   https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf
   If the file is not already in this folder (any .pdf), build_index.py downloads it as
   2025-Budget-Statement-and-Economic-Policy_v4.pdf

   You may also copy the same PDF from your machine (e.g. from Cursor workspaceStorage
   pdfs folder) into data/ — then run: python scripts/build_index.py
