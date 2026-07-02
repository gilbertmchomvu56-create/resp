# Customer Sentiment Complaint Analysis Platform

A Streamlit dashboard for analyzing customer complaints from Tanzanian banks
and mobile money providers (CRDB, NMB, NBC, M-Pesa, Tigo Pesa, Airtel Money,
Mixx by Yas). Includes sentiment classification, topic modeling, authenticity
detection, and exportable reports.

## Features

- Dashboard with 6 KPI cards, donut, bar, line, and regional charts
- Upload CSV datasets with column validation
- Complaint Explorer with multi-filter interactive table
- Sentiment Analysis (rules + scikit-learn) with word cloud
- Topic Analysis by institution and region
- Authenticity Detection (duplicates, spam, suspicious score)
- Reports export as CSV, Excel, and PDF

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this folder to a GitHub repository.
2. Go to https://share.streamlit.io and create a new app.
3. Point it at your repo, branch, and `app.py` as the entry file.
4. Deploy — no extra configuration required.

## Expected CSV columns

`complaint_id, date, institution, complaint_text, sentiment, topic,
authenticity_score, authenticity_label, region, channel, resolution_status`

If no dataset is uploaded, a synthetic sample dataset is generated so every
page renders immediately.

## Project structure

```
.
├── app.py
├── requirements.txt
├── README.md
├── assets/
└── data/
```
