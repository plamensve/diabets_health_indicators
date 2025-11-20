import gradio as gr
import joblib
import pandas as pd

# Load model
model = joblib.load("models/logistic_regression_model.pkl")

# Final column list (ALL REAL + ALL NEW ADDED)
columns = [
    'age', 'gender', 'physical_activity_minutes_per_week', 'diet_score',
    'sleep_hours_per_day', 'screen_time_hours_per_day',
    'family_history_diabetes', 'hypertension_history',
    'cardiovascular_history', 'bmi', 'waist_to_hip_ratio', 'systolic_bp',
    'diastolic_bp', 'heart_rate', 'cholesterol_total', 'hdl_cholesterol',
    'ldl_cholesterol', 'triglycerides', 'glucose_fasting',
    'glucose_postprandial', 'insulin_level', 'hba1c',
    'diabetes_risk_score',

    # ETHNICITY (existing + new)
    'ethnicity_Black', 'ethnicity_Hispanic', 'ethnicity_Other',
    'ethnicity_White', 'ethnicity_Asian',

    # EDUCATION
    'education_level_Highschool', 'education_level_No formal',
    'education_level_Postgraduate',

    # INCOME (existing + new)
    'income_level_Low', 'income_level_Lower-Middle', 'income_level_Middle',
    'income_level_Upper-Middle', 'income_level_High',

    # EMPLOYMENT (existing + new)
    'employment_status_Retired', 'employment_status_Student',
    'employment_status_Unemployed', 'employment_status_Employed',

    # SMOKING (existing + new)
    'smoking_status_Former', 'smoking_status_Never',
    'smoking_status_Current',

    # Alcohol one-hot
    'alcohol_consumption_per_week_1', 'alcohol_consumption_per_week_2',
    'alcohol_consumption_per_week_3', 'alcohol_consumption_per_week_4',
    'alcohol_consumption_per_week_5', 'alcohol_consumption_per_week_6',
    'alcohol_consumption_per_week_7', 'alcohol_consumption_per_week_8',
    'alcohol_consumption_per_week_9', 'alcohol_consumption_per_week_10',
]

# Dropdown options (your updated ones)
ethnicity_opts = ["Black", "Hispanic", "Other", "White", "Asian"]
education_opts = ["Highschool", "No formal", "Postgraduate"]
income_opts = ["Low", "High", "Lower-Middle", "Middle", "Upper-Middle"]
employment_opts = ["Retired", "Student", "Unemployed", "Employed"]
smoking_opts = ["Former", "Never", "Current"]


def predict_fn(
    age, gender, physical_activity, diet_score, sleep_hours, screen_time,
    family_hist, hyper, cardio, bmi, waist_hip, sbp, dbp, hr,
    chol, hdl, ldl, tri, glu_f, glu_p, insulin, hba1c, risk_score,
    ethnicity, education, income, employment, smoking, alcohol
):

    def yesno(x): return 1 if x == "Yes" else 0
    gender_val = 1 if gender == "Male" else 0

    # Base numeric values
    row = [
        age, gender_val, physical_activity, diet_score,
        sleep_hours, screen_time,
        yesno(family_hist), yesno(hyper), yesno(cardio),
        bmi, waist_hip, sbp, dbp, hr,
        chol, hdl, ldl, tri, glu_f, glu_p, insulin, hba1c, risk_score
    ]

    # One-hot ethnicity
    for opt in ethnicity_opts:
        row.append(1 if ethnicity == opt else 0)

    # Education OHE
    for opt in education_opts:
        row.append(1 if education == opt else 0)

    # Income OHE
    for opt in income_opts:
        row.append(1 if income == opt else 0)

    # Employment OHE
    for opt in employment_opts:
        row.append(1 if employment == opt else 0)

    # Smoking OHE
    for opt in smoking_opts:
        row.append(1 if smoking == opt else 0)

    # Alcohol OHE
    for i in range(1, 11):
        row.append(1 if alcohol == i else 0)

    df = pd.DataFrame([row], columns=columns)
    pred = model.predict(df)[0]
    return f"Prediction: {pred}"


# -------------- UI -----------------------

with gr.Blocks() as app:

    gr.Markdown("## ❤️ Diabetes Prediction UI (Logistic Regression Model)")
    gr.Markdown("Enter patient information below. Normal reference values are provided as guidance.")

    # ----------------- PERSONAL SECTION -----------------
    with gr.Accordion("👤 Personal Information", open=True):
        with gr.Row():
            age = gr.Number(label="Age", placeholder="Normal: 18–70")
            gender = gr.Radio(["Male", "Female"], label="Gender")

        with gr.Row():
            ethnicity = gr.Dropdown(ethnicity_opts, label="Ethnicity")
            education = gr.Dropdown(education_opts, label="Education Level")

        with gr.Row():
            income = gr.Dropdown(income_opts, label="Income Level")
            employment = gr.Dropdown(employment_opts, label="Employment Status")

    # ----------------- LIFESTYLE SECTION -----------------
    with gr.Accordion("🏃 Lifestyle Factors", open=True):
        with gr.Row():
            smoking = gr.Dropdown(smoking_opts, label="Smoking Status")
            alcohol = gr.Slider(0, 10, step=1, label="Alcohol Drinks/Week (0 = Normal)")

        with gr.Row():
            physical_activity = gr.Number(
                label="Physical Activity (min/week)",
                placeholder="Normal: ≥150"
            )
            diet_score = gr.Number(
                label="Diet Score (0–10)",
                placeholder="Normal: ≥7"
            )

        with gr.Row():
            sleep_hours = gr.Number(
                label="Sleep Hours/Day",
                placeholder="Normal: 7–9"
            )
            screen_time = gr.Number(
                label="Screen Time/Day (hours)",
                placeholder="Normal: <2"
            )

    # ----------------- MEDICAL SECTION -----------------
    with gr.Accordion("🩺 Medical Information", open=True):

        with gr.Row():
            family_hist = gr.Radio(["No", "Yes"], label="Family History of Diabetes")
            hyper = gr.Radio(["No", "Yes"], label="Hypertension History")

        with gr.Row():
            cardio = gr.Radio(["No", "Yes"], label="Cardiovascular History")
            bmi = gr.Number(label="BMI", placeholder="Normal: 18.5–24.9")

        with gr.Row():
            waist_hip = gr.Number(
                label="Waist-to-Hip Ratio",
                placeholder="Normal: <0.90 (men), <0.85 (women)"
            )
            sbp = gr.Number(
                label="Systolic Blood Pressure",
                placeholder="Normal: 90–120 mmHg"
            )

        with gr.Row():
            dbp = gr.Number(
                label="Diastolic Blood Pressure",
                placeholder="Normal: 60–80 mmHg"
            )
            hr = gr.Number(
                label="Resting Heart Rate",
                placeholder="Normal: 60–100 bpm"
            )

        with gr.Row():
            chol = gr.Number(
                label="Total Cholesterol",
                placeholder="Normal: <200 mg/dL"
            )
            hdl = gr.Number(
                label="HDL Cholesterol",
                placeholder="Normal: >40 (men), >50 (women)"
            )

        with gr.Row():
            ldl = gr.Number(
                label="LDL Cholesterol",
                placeholder="Normal: <100 mg/dL"
            )
            tri = gr.Number(
                label="Triglycerides",
                placeholder="Normal: <150 mg/dL"
            )

        with gr.Row():
            glu_f = gr.Number(
                label="Fasting Glucose",
                placeholder="Normal: 70–99 mg/dL"
            )
            glu_p = gr.Number(
                label="Postprandial Glucose",
                placeholder="Normal: <140 mg/dL"
            )

        with gr.Row():
            insulin = gr.Number(
                label="Insulin Level",
                placeholder="Normal: 2–20 µIU/mL"
            )
            hba1c = gr.Number(
                label="HbA1c (%)",
                placeholder="Normal: <5.7%"
            )

        with gr.Row():
            risk_score = gr.Number(
                label="Diabetes Risk Score",
                placeholder="Typical: <30"
            )

    # ----------------- BUTTON & OUTPUT -----------------
    output = gr.Textbox(label="Prediction Result")
    btn = gr.Button("Predict Diabetes")

    btn.click(
        predict_fn,
        inputs=[
            age, gender, physical_activity, diet_score, sleep_hours, screen_time,
            family_hist, hyper, cardio, bmi, waist_hip, sbp, dbp, hr,
            chol, hdl, ldl, tri, glu_f, glu_p, insulin, hba1c, risk_score,
            ethnicity, education, income, employment, smoking, alcohol
        ],
        outputs=output
    )

app.launch(share=True)
