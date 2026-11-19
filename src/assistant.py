"""
BARBARIK Tactical AI Copilot & Academic Viva Advisor
Intelligent rule-and-knowledge-based assistant for combat scenario analysis,
parameter guidance, and machine learning methodology defense.
"""

from typing import Dict, Any, List, Optional
import re


def generate_tactical_reply(
    message: str,
    scenario: Optional[Dict[str, Any]] = None,
    prediction: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate an intelligent, context-aware tactical advisory reply.
    
    Args:
        message: User query string
        scenario: Currently active scenario parameters from the UI
        prediction: Latest prediction result dictionary from /api/predict
    
    Returns:
        Dict with 'reply' (markdown text) and 'suggested_prompts' (list of strings)
    """
    msg = message.strip().lower()
    
    # -------------------------------------------------------------------------
    # 1. LIVE SCENARIO ANALYSIS & ADVICE
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["analyze", "current scenario", "what is happening", "situation report", "sitrep"]):
        if not scenario:
            return {
                "reply": "⚠️ **No active scenario detected.** Adjust the force sliders on the left and click **RUN SIMULATION** to generate a tactical assessment.",
                "suggested_prompts": ["How can Blue Force win?", "Why did the model choose this outcome?", "Explain XGBoost"]
            }
        
        force_a = scenario.get('side_a_force_strength', 50)
        force_b = scenario.get('side_b_force_strength', 50)
        ratio = force_a / (force_b + 1)
        intel_diff = scenario.get('side_a_intelligence', 0.5) - scenario.get('side_b_intelligence', 0.5)
        logistics_diff = scenario.get('side_a_logistics', 0.5) - scenario.get('side_b_logistics', 0.5)
        terrain = scenario.get('terrain_type', 'Open')
        weather = scenario.get('weather', 'Clear')
        roe = scenario.get('rules_of_engagement', 'Permissive')
        
        outcome = prediction.get('outcome', 'PENDING') if prediction else 'PENDING'
        confidence = prediction.get('confidence', 0.0) if prediction else 0.0
        
        analysis = (
            f"### 🛡️ Tactical Situation Report (SITREP)\n\n"
            f"- **Current Prediction:** `{outcome}` (Confidence: {confidence*100:.1f}%)\n"
            f"- **Force Balance:** Blue Strength `{force_a}` vs Red Strength `{force_b}` (Force Ratio: `{ratio:.2f}`)\n"
            f"- **Information Warfare:** Intelligence margin is `{'+' if intel_diff > 0 else ''}{intel_diff:.2f}` for Blue\n"
            f"- **Logistical Reach:** Logistics margin is `{'+' if logistics_diff > 0 else ''}{logistics_diff:.2f}` for Blue\n"
            f"- **Environment:** `{terrain}` terrain under `{weather}` weather with `{roe}` Rules of Engagement.\n\n"
        )
        
        if outcome == "BLUE FORCE ADVANTAGE":
            analysis += (
                "**Key Drivers:** Blue maintains superior relative force strength, training, or logistics. "
                "The offensive envelope is favorable."
            )
        elif outcome == "RED FORCE ADVANTAGE":
            analysis += (
                "**Key Drivers:** Red dominates through favorable force ratios, better defensive positioning, "
                "or Blue has exhausted its logistics line (check Blue supply distance and fatigue)."
            )
        else:
            analysis += (
                "**Key Drivers:** Neither side holds the decisive **0.15 score threshold**. "
                "Roughly matched force parameters, high environmental difficulty, or 'Delay' objectives "
                "are forcing an operational stalemate."
            )
            
        return {
            "reply": analysis,
            "suggested_prompts": ["How can Blue Force win?", "How can Red Force win?", "Explain SHAP factors"]
        }

    # -------------------------------------------------------------------------
    # 2. HOW CAN BLUE WIN?
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["how can blue win", "blue force win", "make blue win", "blue advantage"]):
        advice = (
            "### 🔵 Recommended Tactical Adjustments for Blue Force Victory:\n\n"
            "1. **Force Ratio**: Increase Blue Force Strength to **≥ 80** while keeping Red below 65 (ideal ratio `> 1.25`).\n"
            "2. **Surprise Factor**: Set Blue Surprise to **≥ 0.70**. A surprise factor above 0.7 adds an immediate **+0.15** decisive bonus.\n"
            "3. **Logistics & Distance**: Keep Blue Supply Distance **< 100 km** and Logistics **≥ 0.75**. High distance degrades effective logistics linearly.\n"
            "4. **Mitigate Fatigue**: Lower Blue Fatigue below **0.25**. Fatigue directly degrades readiness by up to −30% and mobility by −20%.\n"
            "5. **Electronic Warfare**: If EW level is High/Severe, communication drops by up to 60%. Lower EW or increase Blue Training to compensate.\n"
            "6. **Rules of Engagement**: Switch Mission Objective to **Attack** and RoE to **Offensive** for combat bonuses."
        )
        return {
            "reply": advice,
            "suggested_prompts": ["How can Red Force win?", "Analyze active scenario", "Explain Fatigue effect"]
        }

    # -------------------------------------------------------------------------
    # 3. HOW CAN RED WIN?
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["how can red win", "red force win", "make red win", "red advantage"]):
        advice = (
            "### 🔴 Recommended Tactical Adjustments for Red Force Victory:\n\n"
            "1. **Defensive Posture**: Set Objective to **Defend** in **Hilly**, **Forest**, or **Urban** terrain. This grants Red an immediate **+0.10** defense modifier.\n"
            "2. **Force Superiority**: Increase Red Force Strength to **≥ 85** and Equipment Quality to **≥ 0.85**.\n"
            "3. **Overextend Blue**: Increase Blue Supply Distance to **400+ km** and Blue Fatigue to **0.7+**.\n"
            "4. **Rules of Engagement**: Set RoE to **Defensive** (provides +0.07 score bonus to Red and −0.04 penalty to Blue).\n"
            "5. **Electronic Warfare**: Increase EW to **Severe (1.0)** to sever Blue's communication lines."
        )
        return {
            "reply": advice,
            "suggested_prompts": ["How can Blue Force win?", "Why did stalemate happen?", "Explain XGBoost"]
        }

    # -------------------------------------------------------------------------
    # 4. WHY STALEMATE?
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["stalemate", "why draw", "why stalemate", "why is it stalemate"]):
        reply = (
            "### ⚖️ Why Stalemates Occur in BARBARIK:\n\n"
            "In our simulation engine, a decisive advantage is declared only if:\n"
            "$$\\Delta = |\\text{Score}_A - \\text{Score}_B| > \\text{Threshold}$$"
            "\n"
            "- **Standard Decisive Threshold**: `0.15`\n"
            "- If $|\Delta| \\le 0.15$, neither force can achieve operational breakout, resulting in **STALEMATE**.\n"
            "- **Objectives Altering Stalemate Frequency**:\n"
            "  * **Delay**: Drops the threshold to `0.10`, deliberately prolonging symmetric attrition and favoring stalemates.\n"
            "  * **Withdraw**: Increases the threshold to `0.20`, making stalemates rarer.\n"
            "- **Tactical Cause**: When forces have roughly balanced force strengths, comparable equipment, and neutral terrain, the outcome naturally settles into a defensive stalemate (~50% of synthetic scenarios)."
        )
        return {
            "reply": reply,
            "suggested_prompts": ["How can Blue Force win?", "Analyze active scenario", "Explain RoE"]
        }

    # -------------------------------------------------------------------------
    # 5. VIVA & ACADEMIC DEFENSE: XGBOOST
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["xgboost", "why xgboost", "random forest", "why did accuracy increase", "model choice"]):
        reply = (
            "### 🎓 Machine Learning Viva Defense: Why XGBoost?\n\n"
            "1. **Sequential Residual Learning (Boosting vs Bagging)**:\n"
            "   * Random Forest builds independent trees via bootstrap aggregation and averages them.\n"
            "   * XGBoost builds trees sequentially. Each new tree fits to the negative gradient (residuals) of the multi-class log-loss, directly targeting misclassified samples near class boundaries.\n"
            "2. **Handling Diagonal Decision Boundaries**:\n"
            "   * Combat outcomes depend on relative margins ($A - B > c$). With derived features (`force_ratio`, `intel_diff`, `logistics_diff`), XGBoost achieved **90.35% test accuracy** (up from 86.0%).\n"
            "3. **Regularized Objective**:\n"
            "   * XGBoost includes explicit L1 (`reg_alpha=0.1`) and L2 (`reg_lambda=1.0`) leaf weight penalties, preventing deep tree overfitting to stochastic combat noise.\n"
            "4. **Verified Performance**:\n"
            "   * **Test Accuracy:** `90.35%`\n"
            "   * **Multiclass ROC-AUC:** `0.98161`\n"
            "   * **Weighted F1-Score:** `0.90334`"
        )
        return {
            "reply": reply,
            "suggested_prompts": ["Explain SHAP (XAI)", "How was dataset generated?", "What are the 36 features?"]
        }

    # -------------------------------------------------------------------------
    # 6. VIVA & ACADEMIC DEFENSE: SHAP & XAI
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["shap", "xai", "explainability", "why this prediction", "interpretability"]):
        reply = (
            "### 🔍 Explainable AI (XAI) with SHAP TreeExplainer:\n\n"
            "1. **Game-Theoretic Foundation**:\n"
            "   * SHAP (SHapley Additive exPlanations) is based on cooperative game theory (Shapley values).\n"
            "   * It assigns each feature an additive importance score representing its marginal contribution to the prediction compared to the base expected outcome.\n"
            "2. **TreeExplainer Integration**:\n"
            "   * We use `shap.TreeExplainer(model)` which runs in polynomial time $\\mathcal{O}(TLD^2)$ rather than exponential time.\n"
            "   * Computes exact feature attributions across tree ensembles for multi-class log-odds.\n"
            "3. **Local vs Global Explanations**:\n"
            "   * **Local Explanation** (*Why this scenario?*): Visualized in the 'Top Influencing Factors' horizontal bar chart and the XAI drawer.\n"
            "   * **Global Explanation**: The SHAP summary plot ranks the most decisive strategic levers across the entire 20,000 scenario dataset."
        )
        return {
            "reply": reply,
            "suggested_prompts": ["Why XGBoost?", "How was dataset generated?", "Analyze active scenario"]
        }

    # -------------------------------------------------------------------------
    # 7. DATASET & SIMULATION ENGINE METHODOLOGY
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["dataset", "dataset generation", "synthetic", "how is data generated", "noise"]):
        reply = (
            "### 📊 Synthetic Dataset Methodology (20,000 Scenarios):\n\n"
            "1. **Why Synthetic Combat Data?**\n"
            "   * Real-world bilateral tactical military conflict data is scarce, highly classified, and non-standardized.\n"
            "   * A calibrated rule-based simulator ensures complete ground-truth transparency and reproducible scientific experiments.\n"
            "2. **Generation Process**:\n"
            "   * Deterministic seed: `42`.\n"
            "   * 20,000 randomized combat scenarios across 36 features.\n"
            "   * Stochastic Gaussian noise $\\sigma = 0.015$ added to combat scores to represent combat friction and fog of war.\n"
            "3. **Data Quality Diagnostics**:\n"
            "   * Total missing values: `0`\n"
            "   * Exact duplicates: `0`\n"
            "   * Target leakage: `0` (zero overlap across train/test splits)."
        )
        return {
            "reply": reply,
            "suggested_prompts": ["What are the 36 features?", "Why XGBoost?", "Explain SHAP (XAI)"]
        }

    # -------------------------------------------------------------------------
    # 8. THE 36 FEATURES EXPLAINED
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["features", "36 features", "parameters", "derived features", "input variables"]):
        reply = (
            "### 📋 The 36 Features in BARBARIK:\n\n"
            "**1. Core Capabilities (12)**: Force Strength (30-100), Mobility, Logistics, Intelligence, Communication, Readiness (0.3-1.0) for both Blue and Red.\n"
            "**2. Tactical Modifiers (10)**: Morale (0.0-1.0), Training (0.0-1.0), Surprise (0.0-1.0), Fatigue (0.0-1.0), Equipment Quality (0.3-1.0) for both Blue and Red.\n"
            "**3. Logistics Geometry (2)**: Supply Distance (0-500 km) for Blue and Red.\n"
            "**4. Environment & Difficulty (2)**: Electronic Warfare Level (0.0-1.0), Environmental Difficulty (0.0-1.0).\n"
            "**5. Categorical Environment & Mission (7)**: Terrain (6), Weather (5), Visibility (3), Time of Day (2), Objective Type (4), Season (4), Rules of Engagement (4).\n"
            "**6. Derived Interaction Terms (3)**:\n"
            "   - `force_ratio = side_a_force / (side_b_force + 1)`\n"
            "   - `intel_diff = side_a_intel - side_b_intel`\n"
            "   - `logistics_diff = side_a_logistics - side_b_logistics`"
        )
        return {
            "reply": reply,
            "suggested_prompts": ["Why XGBoost?", "Analyze active scenario", "Explain Fatigue effect"]
        }

    # -------------------------------------------------------------------------
    # 9. SPECIFIC PARAMETER EXPLANATIONS (Fatigue, EW, Season, RoE)
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["fatigue", "tired"]):
        return {
            "reply": (
                "### 💤 Fatigue Parameter Mechanics:\n\n"
                "- **Range**: `0.0` (Fresh troops) to `1.0` (Exhausted troops).\n"
                "- **Impact**:\n"
                "  * Readiness Penalty: Multiplied by $\\max(0.5, 1.0 - 0.3 \\times \\text{fatigue})$ (up to **−30%** readiness).\n"
                "  * Mobility Penalty: Multiplied by $\\max(0.6, 1.0 - 0.2 \\times \\text{fatigue})$ (up to **−20%** mobility).\n"
                "- **Tactical Advice**: High fatigue cripples an attacking force even if raw numbers are superior."
            ),
            "suggested_prompts": ["Explain Equipment Quality", "Explain Electronic Warfare", "How can Blue Force win?"]
        }

    if any(k in msg for k in ["equipment", "gear", "weapon"]):
        return {
            "reply": (
                "### 🎖️ Equipment Quality Parameter Mechanics:\n\n"
                "- **Range**: `0.3` (Outdated gear) to `1.0` (Modern elite gear).\n"
                "- **Modifier Formula**: $\\text{equip\\_mod} = 0.75 + 0.40 \\times \\text{equipment\\_quality}$ (scales from **0.75×** to **1.15×**).\n"
                "- **Impact**: Directly multiplies Force Strength contribution and effective tactical mobility."
            ),
            "suggested_prompts": ["Explain Fatigue effect", "Explain Electronic Warfare", "Analyze active scenario"]
        }

    if any(k in msg for k in ["electronic warfare", "ew", "jamming"]):
        return {
            "reply": (
                "### 📡 Electronic Warfare (EW) Parameter Mechanics:\n\n"
                "- **Levels**: None (`0.0`), Low (`0.25`), Moderate (`0.50`), High (`0.75`), Severe (`1.0`).\n"
                "- **Formula**: $\\text{effective\\_comm} = \\text{communication} \\times (1.0 - 0.60 \\times \\text{EW\\_level})$.\n"
                "- **Impact**: Severe EW suppresses communication efficiency by up to **60%**, disrupting command-and-control synchronization."
            ),
            "suggested_prompts": ["Explain Rules of Engagement", "Explain Season effect", "How can Blue Force win?"]
        }

    if any(k in msg for k in ["rules of engagement", "roe"]):
        return {
            "reply": (
                "### ⚖️ Rules of Engagement (RoE) Parameter Mechanics:\n\n"
                "- **Permissive**: Unrestricted tactical engagement; both forces operate at 100% capacity.\n"
                "- **Restrictive**: Constrained fire; reduces communication & coordination efficiency by 15%.\n"
                "- **Defensive**: Prioritizes territorial retention; awards **+0.07 score bonus** to defending Red Force and **−0.04 penalty** to Blue Force.\n"
                "- **Offensive**: Authorizes pre-emptive strikes; awards **+0.07 score bonus** to attacking Blue Force and **−0.04 penalty** to Red Force."
            ),
            "suggested_prompts": ["Explain Objective Type", "Explain Season effect", "How can Blue Force win?"]
        }

    # -------------------------------------------------------------------------
    # 10. KEYBOARD SHORTCUTS & NAVIGATION
    # -------------------------------------------------------------------------
    if any(k in msg for k in ["shortcut", "hotkey", "keys", "commands", "controls"]):
        return {
            "reply": (
                "### ⌨️ Tactical Console Keyboard Shortcuts:\n\n"
                "- `R` : **Run Simulation** (Execute inference pipeline)\n"
                "- `N` : **New Scenario** (Generate fresh scenario ID)\n"
                "- `E` : **Explainability** (Open 'Why This Prediction?' SHAP drawer)\n"
                "- `P` : **Presentation Mode** (Toggle clean fullscreen HUD)\n"
                "- `D` : **Demo Mode** (Trigger automated demo cycle)\n"
                "- `S` : **Audio Toggle** (Mute / unmute tactical electronic sound effects)\n"
                "- `ESC` : **Dismiss** (Close all active drawers, modals, or chat window)"
            ),
            "suggested_prompts": ["Analyze active scenario", "Why XGBoost?", "How can Blue Force win?"]
        }

    # -------------------------------------------------------------------------
    # 11. DEFAULT FALLBACK WITH CAPABILITIES
    # -------------------------------------------------------------------------
    return {
        "reply": (
            f"### 🤖 BARBARIK Tactical Copilot Ready\n\n"
            f"I am your real-time wargaming advisor and ML methodology assistant. You can ask me:\n\n"
            f"1. **Scenario Advice**: *'Analyze the current scenario'*, *'How can Blue win?'*, *'Why did Stalemate occur?'*\n"
            f"2. **Parameter Impact**: *'Explain Fatigue'*, *'How does Electronic Warfare work?'*, *'Explain Rules of Engagement'*\n"
            f"3. **Viva Defense**: *'Why XGBoost instead of Random Forest?'*, *'How does SHAP explainability work?'*, *'How was the dataset generated?'*\n"
            f"4. **Controls**: *'What are the keyboard shortcuts?'*"
        ),
        "suggested_prompts": ["Analyze active scenario", "How can Blue Force win?", "Why XGBoost?", "Explain SHAP (XAI)"]
    }
