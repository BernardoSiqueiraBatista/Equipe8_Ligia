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
    
    # Adiciona ao sys.modules para poder importar depois
    sys.modules["counterfactual"] = counterfactual_module
    
    # Executa o módulo
    spec.loader.exec_module(counterfactual_module)
    
    # Pega as funções
    generate_counterfactual = counterfactual_module.generate_counterfactual
    analyze_habits = counterfactual_module.analyze_habits
    format_analysis_report = counterfactual_module.format_analysis_report
    
  
    
except Exception as e:

    
    # Método 2: Executa o arquivo diretamente
    try:
        # Lê o conteúdo do arquivo e executa
        with open(counterfactual_file, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Cria um namespace local
        local_vars = {}
        exec(code, globals(), local_vars)
        
        # Pega as funções do namespace
        generate_counterfactual = local_vars['generate_counterfactual']
        analyze_habits = local_vars['analyze_habits']
        format_analysis_report = local_vars['format_analysis_report']
       
        
    except Exception as e2:
        st.error(f"❌ Todos os métodos falharam: {e2}")
        st.stop()

# 4. CARREGA O MODELO
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
    age = st.number_input("Sua Idade:", key="age")
    gender = st.text_input("Seu Gênero:")
    occupation = st.text_input("Sua Profissão", key="occupation")
    sleep_duration = st.number_input("Sua Duração de Sono", key="sleep_duration")
    physical_activity_level = st.number_input("Seu Nível de Atividade Física", key="physical_activity_level")
    stress_level = st.number_input("Seu Nível de Estresse", key="stress_level")
    daily_steps = st.number_input("Seus Passos Diários", key="daily_steps")
    systolic_bp = st.number_input("Sua Pressão Sistólica", key="systolic_bp")
    diastolic_bp = st.number_input("Sua Pressão Diastólica", key="diastolic_bp")
    sleep_disorder = st.text_input("Qual distúrbio do sono você possui? Nenhum/Insônia/", key="sleep_disorder")
    bmi_category = st.text_input("Qual sua categoria de IMC:", key="bmi_category")
    heart_rate = st.number_input("Qual sua Heart Rate", key="heart_rate")

    submitted = st.form_submit_button("Analisar Hábitos de Saúde")

if submitted:
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

    st.write("## Análise dos seus hábitos de saúde:")
    st.write(format_analysis_report(analysis))
    st.write("\nSugestão de contraexemplo:")
    st.write(generate_counterfactual(user_input, model))



