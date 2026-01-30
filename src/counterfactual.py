import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Features que podem ser modificadas pelo usuário
EDITABLE_FEATURES = [
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Daily Steps",
]

# Valores de referência para identificar hábitos problemáticos
REFERENCE_VALUES = {
    "Sleep Duration": {
        "ideal_min": 7.0,
        "ideal_max": 9.0,
        "unit": "horas",
        "direction": "higher_better",
        "explanation": "Adultos devem dormir entre 7-9 horas por noite para uma saúde ótima.",
        "tip": "Estabeleça um horário regular para dormir e acordar, mesmo nos fins de semana."
    },
    "Physical Activity Level": {
        "ideal_min": 60,
        "ideal_max": 90,
        "unit": "minutos/dia",
        "direction": "higher_better",
        "explanation": "A atividade física regular melhora a qualidade do sono e reduz o estresse.",
        "tip": "Tente fazer pelo menos 30-60 minutos de atividade física moderada diariamente."
    },
    "Stress Level": {
        "ideal_min": 1,
        "ideal_max": 4,
        "unit": "escala 1-10",
        "direction": "lower_better",
        "explanation": "Níveis altos de estresse prejudicam significativamente a qualidade do sono.",
        "tip": "Pratique técnicas de relaxamento como meditação, respiração profunda ou yoga."
    },
    "Daily Steps": {
        "ideal_min": 7000,
        "ideal_max": 10000,
        "unit": "passos",
        "direction": "higher_better",
        "explanation": "Caminhar regularmente melhora a saúde cardiovascular e a qualidade do sono.",
        "tip": "Use escadas, caminhe durante ligações telefônicas e faça pausas ativas no trabalho."
    },
    "Heart Rate": {
        "ideal_min": 60,
        "ideal_max": 75,
        "unit": "bpm",
        "direction": "lower_better",
        "explanation": "Frequência cardíaca de repouso elevada pode indicar estresse ou falta de condicionamento.",
        "tip": "Exercícios aeróbicos regulares ajudam a reduzir a frequência cardíaca de repouso."
    }
}


def _predict_good_sleep_proba(model: Pipeline, x: pd.DataFrame) -> float:
    """Retorna P(classe positiva) assumindo predict_proba com coluna 1."""
    proba = model.predict_proba(x)
    return float(proba[0, 1])


def _processed_feature_scores(clf, feature_names):
    """
    Extrai um score por feature (já após o preprocessor) de forma flexível.

    - LogisticRegression: usa coef_ (direcional)
    - RandomForestClassifier: usa feature_importances_ (não direcional)
    - MLPClassifier: aproxima importância pela soma |pesos| da 1a camada (não direcional)
    """
    if isinstance(clf, LogisticRegression) and hasattr(clf, "coef_"):
        try:
            coefs = np.asarray(clf.coef_)
            if coefs.ndim == 2:
                coefs = coefs[0]
            return {feature_names[i]: float(coefs[i]) for i in range(min(len(feature_names), len(coefs)))}
        except Exception:
            pass

    if isinstance(clf, RandomForestClassifier) and hasattr(clf, "feature_importances_"):
        try:
            imps = np.asarray(clf.feature_importances_)
            return {feature_names[i]: float(imps[i]) for i in range(min(len(feature_names), len(imps)))}
        except Exception:
            pass

    if isinstance(clf, MLPClassifier) and hasattr(clf, "coefs_"):
        # MLPClassifier: coefs_[0] tem shape (n_features, n_hidden)
        try:
            w0 = np.asarray(clf.coefs_[0])
            imp = np.sum(np.abs(w0), axis=1)
            return {feature_names[i]: float(imp[i]) for i in range(min(len(feature_names), len(imp)))}
        except Exception:
            pass

    return {}


def _aggregate_raw_feature_score(processed_scores: dict, raw_feature: str) -> float:
    """Agrega scores de features processadas (ex: one-hot) em uma feature original."""
    total = 0.0
    for fname, score in processed_scores.items():
        if raw_feature in fname:
            total += float(score)
    return float(total)


def _infer_search_params(feature: str):
    """Heurísticas de step e limites para busca model-agnostic."""
    ref = REFERENCE_VALUES.get(feature, {})
    ideal_min = float(ref.get("ideal_min", 0))
    ideal_max = float(ref.get("ideal_max", ideal_min + 1))

    if feature == "Sleep Duration":
        step = 0.25
        bounds = (0.0, max(12.0, ideal_max * 1.5))
    elif feature == "Physical Activity Level":
        step = 10.0
        bounds = (0.0, max(180.0, ideal_max * 2.0))
    elif feature == "Stress Level":
        step = 1.0
        bounds = (1.0, 10.0)
    elif feature == "Daily Steps":
        step = 500.0
        bounds = (0.0, max(20000.0, ideal_max * 2.0))
    else:
        # caso genérico
        step = max(1.0, (ideal_max - ideal_min) / 10.0)
        bounds = (0.0, max(ideal_max * 2.0, ideal_min + 10.0))

    return step, bounds


def _best_single_feature_move(
    x_original: pd.DataFrame,
    model: Pipeline,
    feature: str,
    threshold: float,
    max_steps: int = 40,
    feature_bounds=None,
):
    """Busca a melhor mudança em UMA feature para ultrapassar o threshold.

    Se não cruzar o limiar, retorna a melhor melhora encontrada.
    """
    if feature not in x_original.columns:
        return None

    current = float(x_original[feature].values[0])
    step, (lo_default, hi_default) = _infer_search_params(feature)
    lo, hi = (lo_default, hi_default)
    if feature_bounds and feature in feature_bounds:
        lo, hi = feature_bounds[feature]

    base_prob = _predict_good_sleep_proba(model, x_original)

    # Escolhe direção que mais aumenta a probabilidade (gradiente por diferença finita)
    eps = step
    prob_up = None
    prob_dn = None

    if current + eps <= hi:
        x_up = x_original.copy()
        x_up.loc[:, feature] = current + eps
        prob_up = _predict_good_sleep_proba(model, x_up)

    if current - eps >= lo:
        x_dn = x_original.copy()
        x_dn.loc[:, feature] = current - eps
        prob_dn = _predict_good_sleep_proba(model, x_dn)

    direction = None
    if prob_up is not None and prob_dn is not None:
        direction = 1 if (prob_up - base_prob) >= (prob_dn - base_prob) else -1
    elif prob_up is not None:
        direction = 1
    elif prob_dn is not None:
        direction = -1
    else:
        return None

    # Caminha na direção escolhida até cruzar threshold
    best = None
    best_improvement = None
    best_prob = base_prob
    for k in range(1, max_steps + 1):
        candidate = current + direction * step * k
        if candidate < lo or candidate > hi:
            break
        x_new = x_original.copy()
        x_new.loc[:, feature] = candidate
        p = _predict_good_sleep_proba(model, x_new)
        if p > best_prob:
            best_prob = p
            delta = abs(candidate - current)
            best_improvement = {
                "feature": feature,
                "from": current,
                "to": float(candidate),
                "delta": float(delta),
                "prob": float(p),
                "steps": k,
                "direction": direction,
                "crossed": bool(p >= threshold),
            }

        if p >= threshold:
            best = best_improvement
            break

    return best if best is not None else best_improvement


def _try_logistic_counterfactual_raw_units(
    x_original: pd.DataFrame,
    model: Pipeline,
    threshold: float,
):
    """Contraexemplo mais interpretável para LogisticRegression em unidades originais.

    Usa o logit alvo: logit(t) = ln(t/(1-t)) e a função de decisão atual.
    Converte delta no espaço padronizado para delta em unidades originais via StandardScaler.
    """
    try:
        preprocessor = model.named_steps["preprocessor"]
        clf = model.named_steps["model"]
        if not (hasattr(clf, "coef_") and hasattr(clf, "decision_function")):
            return None

        # Colunas numéricas e scaler
        num_transformer = preprocessor.named_transformers_.get("num")
        if not isinstance(num_transformer, Pipeline):
            return None
        scaler = num_transformer.named_steps.get("scaler")
        if not hasattr(scaler, "scale_"):
            return None

        num_cols = None
        for name, _t, cols in preprocessor.transformers_:
            if name == "num":
                num_cols = list(cols)
                break
        if not num_cols:
            return None

        x_proc = preprocessor.transform(x_original)
        current_logit = float(np.ravel(clf.decision_function(x_proc))[0])
        target_logit = float(np.log(threshold / (1.0 - threshold)))
        delta_logit = target_logit - current_logit
        if abs(delta_logit) < 1e-12:
            return None

        feature_names = get_feature_names_from_column_transformer(preprocessor)
        coefs = np.asarray(clf.coef_)[0]

        moves = []
        for raw_feature in EDITABLE_FEATURES:
            if raw_feature not in x_original.columns:
                continue
            if raw_feature not in num_cols:
                continue

            # índice da feature no vetor processado (para num passthrough/scaler, o nome é igual)
            try:
                idx = feature_names.index(raw_feature)
            except ValueError:
                continue
            if idx >= len(coefs):
                continue

            w_i = float(coefs[idx])
            if abs(w_i) < 1e-12:
                continue

            # mudança necessária no espaço padronizado
            delta_scaled = delta_logit / w_i
            # converter para unidades originais: x_scaled = (x - mean) / scale
            # então delta_raw = delta_scaled * scale
            scale = float(scaler.scale_[num_cols.index(raw_feature)])
            delta_raw = float(delta_scaled * scale)

            current_raw = float(x_original[raw_feature].values[0])
            target_raw = current_raw + delta_raw

            # clamp em limites razoáveis
            _step, (lo, hi) = _infer_search_params(raw_feature)
            target_raw = float(np.clip(target_raw, lo, hi))
            delta_raw = float(target_raw - current_raw)
            if abs(delta_raw) < 1e-12:
                continue

            moves.append({
                "feature": raw_feature,
                "from": current_raw,
                "to": target_raw,
                "delta": abs(delta_raw),
                "direction": 1 if delta_raw > 0 else -1,
            })

        if not moves:
            return None

        best = min(moves, key=lambda m: m["delta"])
        unit = REFERENCE_VALUES.get(best["feature"], {}).get("unit", "")
        action = "aumente" if best["direction"] > 0 else "reduza"
        return (
            f"Para melhorar o sono, {action} {best['feature']} de {best['from']:.2f} para {best['to']:.2f} {unit} "
            f"(Δ≈{best['delta']:.2f})."
        )
    except Exception:
        return None

def get_feature_names_from_column_transformer(ct: ColumnTransformer):
    """
    Extrai nomes das features de um ColumnTransformer.
    Compatível com scikit-learn < 1.0 (sem get_feature_names_out).
    """
    feature_names = []

    for name, transformer, columns in ct.transformers_:
        if transformer == "drop":
            continue

        if transformer == "passthrough":
            feature_names.extend(columns)
            continue

        # Se for Pipeline, pega o último step
        if isinstance(transformer, Pipeline):
            final_estimator = transformer.steps[-1][1]
        else:
            final_estimator = transformer

        # Tenta get_feature_names_out (só funciona em sklearn < 1.0)
        if hasattr(final_estimator, "get_feature_names_out"):
            try:
                names = final_estimator.get_feature_names_out(columns)
                feature_names.extend(names)
                continue
            except Exception:
                pass

        # Tenta get_feature_names (para sklearn < 1.0, OneHotEncoder)
        if hasattr(final_estimator, "get_feature_names"):
            try:
                names = final_estimator.get_feature_names(columns)
                feature_names.extend(names)
                continue
            except Exception:
                pass

        # StandardScaler e outros transformadores que não alteram nomes
        if isinstance(final_estimator, StandardScaler):
            feature_names.extend(columns)
            continue

        # OneHotEncoder (acessa categories_ diretamente)
        if isinstance(final_estimator, OneHotEncoder):
            for i, col in enumerate(columns):
                for cat in final_estimator.categories_[i]:
                    feature_names.append(f"{col}_{cat}")
            continue

        # Se nenhum método funcionou, usa os nomes das colunas originais
        feature_names.extend(columns)

    return feature_names

def generate_counterfactual(x_original: pd.DataFrame, model: Pipeline, threshold=0.5):
    """
    Retorna a menor mudança acionável para mudar a predição para 'sono bom'
    """
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["model"]

    prob = _predict_good_sleep_proba(model, x_original)

    if prob >= threshold:
        return "Perfil já classificado como sono bom."

    feature_names = get_feature_names_from_column_transformer(preprocessor)

    # Caminho interpretável para LogisticRegression (ou linear com scaler)
    logistic_msg = _try_logistic_counterfactual_raw_units(x_original=x_original, model=model, threshold=threshold)
    if logistic_msg is not None:
        return logistic_msg

    # Outros modelos (RandomForestClassifier, MLPClassifier, ...)
    candidates = []
    for raw_feature in EDITABLE_FEATURES:
        move = _best_single_feature_move(
            x_original=x_original,
            model=model,
            feature=raw_feature,
            threshold=threshold,
        )
        if move is not None:
            candidates.append(move)

    if not candidates:
        return "Nenhuma mudança acionável encontrada (modelo não suporta predict_proba ou features ausentes)."

    crossed = [m for m in candidates if m.get("crossed")]
    best = min(crossed, key=lambda m: m["delta"]) if crossed else max(candidates, key=lambda m: m["prob"])
    unit = REFERENCE_VALUES.get(best["feature"], {}).get("unit", "")
    verb = "ajuste"
    if best.get("crossed"):
        return (
            f"Para melhorar o sono, {verb} {best['feature']} de {best['from']:.2f} para {best['to']:.2f} {unit} "
            f"(Δ≈{best['delta']:.2f})."
        )

    return (
        f"Nenhuma mudança única cruzou o limiar; melhor melhora encontrada: {verb} {best['feature']} de "
        f"{best['from']:.2f} para {best['to']:.2f} {unit} (Δ≈{best['delta']:.2f}), "
        f"probabilidade estimada de sono bom ≈ {best['prob']:.1%}."
    )


def analyze_habits(x_original: pd.DataFrame, model: Pipeline, threshold=0.5):
    """
    Analisa todos os hábitos do usuário, identifica os problemáticos,
    explica por que são problemáticos e sugere melhorias.
    
    Retorna um dicionário com:
    - prediction: predição atual
    - probability: probabilidade de sono bom
    - problematic_habits: lista de hábitos problemáticos com explicações
    - recommendations: lista de recomendações priorizadas
    - model_insights: insights do modelo sobre cada feature
    """
    preprocessor = model.named_steps["preprocessor"]
    clf = model.named_steps["model"]

    prob = _predict_good_sleep_proba(model, x_original)
    label = "Bom" if prob >= threshold else "Ruim"
    
    feature_names = get_feature_names_from_column_transformer(preprocessor)
    processed_scores = _processed_feature_scores(clf, feature_names)
    
    problematic_habits = []
    good_habits = []
    
    # Analisar cada feature editável
    for feature in EDITABLE_FEATURES:
        if feature not in x_original.columns:
            continue
            
        current_value = x_original[feature].values[0]
        ref = REFERENCE_VALUES.get(feature, {})
        
        if not ref:
            continue
        
        ideal_min = ref.get("ideal_min", 0)
        ideal_max = ref.get("ideal_max", float('inf'))
        direction = ref.get("direction", "higher_better")
        unit = ref.get("unit", "")
        explanation = ref.get("explanation", "")
        tip = ref.get("tip", "")
        
        # Determinar se o hábito está dentro do ideal
        is_problematic = False
        severity = "low"  # low, medium, high
        issue = ""
        suggestion = ""
        
        if direction == "higher_better":
            if current_value < ideal_min:
                is_problematic = True
                deficit = ideal_min - current_value
                deficit_pct = (deficit / ideal_min) * 100 if ideal_min > 0 else 0
                
                if deficit_pct > 50:
                    severity = "high"
                elif deficit_pct > 25:
                    severity = "medium"
                else:
                    severity = "low"
                
                issue = f"Valor atual ({current_value:.1f} {unit}) está abaixo do recomendado ({ideal_min}-{ideal_max} {unit})"
                suggestion = f"Aumentar para pelo menos {ideal_min} {unit}"
            else:
                good_habits.append({
                    "feature": feature,
                    "current_value": current_value,
                    "status": "✅ Dentro do recomendado",
                    "unit": unit
                })
                
        elif direction == "lower_better":
            if current_value > ideal_max:
                is_problematic = True
                excess = current_value - ideal_max
                excess_pct = (excess / ideal_max) * 100 if ideal_max > 0 else 0
                
                if excess_pct > 50:
                    severity = "high"
                elif excess_pct > 25:
                    severity = "medium"
                else:
                    severity = "low"
                
                issue = f"Valor atual ({current_value:.1f} {unit}) está acima do recomendado ({ideal_min}-{ideal_max} {unit})"
                suggestion = f"Reduzir para no máximo {ideal_max} {unit}"
            else:
                good_habits.append({
                    "feature": feature,
                    "current_value": current_value,
                    "status": "✅ Dentro do recomendado",
                    "unit": unit
                })
        
        if is_problematic:
            # Encontrar o impacto do modelo para esta feature (pode ser coef, importância, etc.)
            model_impact = _aggregate_raw_feature_score(processed_scores, feature)
            
            problematic_habits.append({
                "feature": feature,
                "current_value": current_value,
                "ideal_range": f"{ideal_min}-{ideal_max} {unit}",
                "severity": severity,
                "issue": issue,
                "explanation": explanation,
                "suggestion": suggestion,
                "tip": tip,
                "model_impact": model_impact
            })
    
    # Ordenar por severidade (high > medium > low)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    problematic_habits.sort(key=lambda x: severity_order.get(x["severity"], 3))
    
    # Gerar recomendações priorizadas
    recommendations = []
    for i, habit in enumerate(problematic_habits, 1):
        severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(habit["severity"], "⚪")
        recommendations.append({
            "priority": i,
            "severity_emoji": severity_emoji,
            "feature": habit["feature"],
            "action": habit["suggestion"],
            "tip": habit["tip"]
        })
    
    return {
        "prediction": label,
        "probability": prob,
        "threshold": threshold,
        "problematic_habits": problematic_habits,
        "good_habits": good_habits,
        "recommendations": recommendations,
        "total_issues": len(problematic_habits)
    }


def format_analysis_report(analysis: dict) -> str:
    """
    Formata a análise em um relatório legível.
    """
    lines = []
    
    # Cabeçalho
    lines.append("=" * 60)
    lines.append("📊 RELATÓRIO DE ANÁLISE DE HÁBITOS DE SONO")
    lines.append("=" * 60)
    
    # Resultado da predição
    emoji = "😊" if analysis["prediction"] == "Bom" else "😟"
    lines.append(f"\n🎯 Qualidade do Sono Prevista: {analysis['prediction']} {emoji}")
    lines.append(f"   Probabilidade de sono bom: {analysis['probability']:.1%}")
    
    # Hábitos problemáticos
    if analysis["problematic_habits"]:
        lines.append(f"\n  HÁBITOS PROBLEMÁTICOS IDENTIFICADOS ({analysis['total_issues']})")
        lines.append("-" * 60)
        
        for habit in analysis["problematic_habits"]:
            severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(habit["severity"], "⚪")
            lines.append(f"\n{severity_emoji} {habit['feature']}")
            lines.append(f"    Situação: {habit['issue']}")
            lines.append(f"    Por quê: {habit['explanation']}")
            lines.append(f"    Sugestão: {habit['suggestion']}")
            lines.append(f"    Dica: {habit['tip']}")
    
    # Hábitos bons
    if analysis["good_habits"]:
        lines.append(f"\n HÁBITOS SAUDÁVEIS")
        lines.append("-" * 60)
        for habit in analysis["good_habits"]:
            lines.append(f"   {habit['status']} {habit['feature']}: {habit['current_value']:.1f} {habit['unit']}")
    
    # Recomendações priorizadas
    if analysis["recommendations"]:
        lines.append(f"\n RECOMENDAÇÕES (por prioridade)")
        lines.append("-" * 60)
        for rec in analysis["recommendations"]:
            lines.append(f"   {rec['priority']}. {rec['severity_emoji']} {rec['feature']}: {rec['action']}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)
