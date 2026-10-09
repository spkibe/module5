# Netflix Churn Stakeholder Dashboard

This Streamlit app supports the BAN6800 Module 5 stakeholder presentation.

## Required model artifact

Place your trained model here before deployment:

```text
models/churn_model_calibrated.joblib
```

If the model is missing, the dashboard still displays performance, fairness, robustness and transparency views, but the live prediction tab will show a warning.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy

1. Push this folder to GitHub.
2. Add `models/churn_model_calibrated.joblib` to the repo or configure secure artifact loading.
3. Go to Streamlit Cloud.
4. Select the repository and set the main file to `streamlit_app.py`.
5. Copy the public URL into Slide 13.
