import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import os
from pathlib import Path
import joblib
import importlib.util

current_file = Path(__file__).absolute()  
app_dir = current_file.parent             
project_root = app_dir.parent             

src_path = project_root / "src"
counterfactual_file = src_path / "counterfactual.py"


if not counterfactual_file.exists():
    st.error(f"❌ Arquivo não encontrado: {counterfactual_file}")
    st.stop()


try:
 
    spec = importlib.util.spec_from_file_location(
        "counterfactual", 
        str(counterfactual_file)
    )
    counterfactual_module = importlib.util.module_from_spec(spec)
    
    
    sys.modules["counterfactual"] = counterfactual_module
    
 
    spec.loader.exec_module(counterfactual_module)
    
  
    generate_counterfactual = counterfactual_module.generate_counterfactual
    analyze_habits = counterfactual_module.analyze_habits
    format_analysis_report = counterfactual_module.format_analysis_report
    
  
    
except Exception as e:

    
    try:
       
        with open(counterfactual_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        
        local_vars = {}
        exec(code, globals(), local_vars)
        
     
        generate_counterfactual = local_vars['generate_counterfactual']
        analyze_habits = local_vars['analyze_habits']
        format_analysis_report = local_vars['format_analysis_report']
       
        
    except Exception as e2:
        st.error(f"❌ Todos os métodos falharam: {e2}")
        st.stop()

model_path = project_root / "artifacts" / "rf.joblib"

if model_path.exists():
    try:
        model = joblib.load(str(model_path))
    except Exception as e:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
else:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()

with st.form("user_input_form"):
    name = st.text_input("Seu Nome:", key="name")
    age = st.text_input("Sua Idade:", key="age")

    gender = st.selectbox(
        "Seu Gênero:",
        options=["Male", "Female"]
    )
    
    occupation = st.selectbox("Informe sua profissão?", options=["None"] + [
                "Software Engineer", "Teacher", "Doctor", "Accountant",
                "Sales Representative", "Marketing Manager", "Data Analyst",
                "HR Manager", "Designer", "Architect"
            ]
        )


    sleep_duration = st.text_input("Sua Duração de Sono", key="sleep_duration")
    physical_activity_level = st.text_input("Seu Nível de Atividade Física [Minutos no dia]: ", key="physical_activity_level")
    stress_level = st.text_input("Seu Nível de Estresse ['Escala 1-10']", key="stress_level")
    daily_steps = st.text_input("Seus Passos Diários", key="daily_steps")
    systolic_bp = st.text_input("Sua Pressão Sistólica", key="systolic_bp")
    diastolic_bp = st.text_input("Sua Pressão Diastólica", key="diastolic_bp")
    sleep_disorder = st.selectbox("Qual distúrbio do sono você possui? ", options=["None", "Sleep Apnea", "Insomnia"], key="sleep_disorder")
    bmi_category = st.selectbox("Qual sua categoria de IMC:", options=["Normal", "Overweight", "Obese"], key="bmi_category")
    heart_rate = st.text_input("Qual sua Heart Rate", key="heart_rate")

    submitted = st.form_submit_button("Analisar Hábitos de Saúde")

if submitted:
    try:
        age = float(age)
        sleep_duration = float(sleep_duration)
        physical_activity_level = float(physical_activity_level)
        stress_level = float(stress_level)
        daily_steps = float(daily_steps)
        systolic_bp = float(systolic_bp)
        diastolic_bp = float(diastolic_bp)
        heart_rate = float(heart_rate)
    except ValueError:
        st.error("Por favor, preencha todos os campos numéricos corretamente.")
        st.stop()

    st.write(f"Analisando hábitos de saúde para {name}...")

    user_input = pd.DataFrame([{
        "Age": age,
        "Sleep Duration": sleep_duration,
        "Physical Activity Level": physical_activity_level,
        "Stress Level": stress_level,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Gender": gender,
        "Occupation": occupation,
        "BMI Category": bmi_category,
        "Sleep Disorder": sleep_disorder,
        "Systolic_BP": systolic_bp,
        "Diastolic_BP": diastolic_bp
    }])
    numeric_features = [
        "Age",
        "Sleep Duration",
        "Physical Activity Level",
        "Stress Level",
        "Heart Rate",
        "Daily Steps",
        'Systolic_BP',
        'Diastolic_BP'
    ]

    categorical_features = [
        "Gender",
        "Occupation",
        "BMI Category",
        "Sleep Disorder"
    ]

    user_input[numeric_features] = user_input[numeric_features].astype(float)
    user_input[categorical_features] = user_input[categorical_features].astype(str) 

    # Análise completa dos hábitos + sugestão de contraexemplo
    model = joblib.load(project_root / "artifacts" / "rf.joblib")
    analysis = analyze_habits(user_input, model)
    st.write(format_analysis_report(analysis))
    st.write("\nSugestão de contraexemplo:")
    st.write(generate_counterfactual(user_input, model))



