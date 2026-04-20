\
# PUBG WinPlacePerc Web App (Streamlit, English UI + Notebook Visuals)

## 1) Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Run the app

```bash
streamlit run app.py
```

Open the printed URL (usually http://localhost:8501).

## 3) Notebook visuals

- Put your final notebook file next to `app.py` and name it: `notebook_report.ipynb`
- The app will render its outputs (charts/tables) on the **Notebook Visuals** page.
- If outputs are missing, run the notebook and save it **with outputs**.

## 4) Offline training (optional)

```bash
python train.py --train train.csv --model_out artifacts/model.joblib
```

Then in the app:
- Go to **Prediction Only**
- Upload `model.joblib` and `test.csv`
