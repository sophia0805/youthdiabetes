#from copyreg import pickle

import shutil
from flask import Flask, redirect, request, jsonify, render_template
from flask_cors import CORS
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
import openai
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY from .env


#import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)
CORS(app, origins=["https://www.youthdiabetes.ai", "https://youthdiabetes-ai.onrender.com"])  # Enable CORS for frontend requests

from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline, FunctionTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from flask import redirect


openai.api_key = os.getenv("OPENAI_API_KEY")
model1 = None


if not os.path.exists('/var/data/youthdiabetes_logisticL1_scoring_bundle.pkl'):
        shutil.copy('youthdiabetes_logisticL1_scoring_bundle.pkl', '/var/data/youthdiabetes_logisticL1_scoring_bundle.pkl')

@app.route('/')
def home():
 return render_template('index.html')

@app.route('/risk')
def report():
 return render_template('risk.html')

@app.route('/resources')
def contact():
 return render_template('resources.html')

@app.route('/about')
def about():
 return render_template('about.html')


# Scoring code for website inputs (Python backend / notebook)
# A) Helpers + robust mapping (matches the website dropdowns)
def midpoint_from_range(s: str) -> float:
    """
    Examples from website:
      '151-170' -> 160.5
      '0.6-0.75' -> 0.675
      '300+' -> 300
      '86+' -> 86
      '0' -> 0
    """
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == "" or s.lower().startswith("select"):
        return np.nan
    if s.endswith("+"):
        return float(s[:-1])
    if "-" in s:
        a, b = s.split("-", 1)
        return (float(a) + float(b)) / 2.0
    return float(s)

def lb_to_kg(lb: float) -> float:
    return lb * 0.45359237

def inch_to_cm(inch: float) -> float:
    return inch * 2.54

def map_gender_to_RIAGENDR(gender: str) -> float:
    g = str(gender).strip().lower()
    return 0.0 if g == "male" else 1.0

def map_race_to_RaceEth(race: str) -> float:
    """
    Website options (exact): Hispanic, Non Hispanic White, Non Hispanic Black, Other :contentReference[oaicite:2]{index=2}
    IMPORTANT: don't use 'if "hispanic" in ...' without checking 'non hispanic' first.
    """
    r = str(race).strip().lower()

    # exact / safest mapping first
    if r in ["non hispanic white", "non-hispanic white"]:
        return 1.0
    if r in ["non hispanic black", "non-hispanic black"]:
        return 2.0
    if r == "hispanic":
        return 0.0
    return 3.0  # Other

def yesno_to_01(x: str) -> int:
    return 1 if str(x).strip().lower() == "yes" else 0

# =====================================
# Outlier Clipper (IQR-based)
# =====================================
class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}

        for col in X_df.columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)

        for col, (low, high) in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=low, upper=high)

        return X_df.values

# =====================================
# Outlier Clipper (IQR-based)
# =====================================
class IQRClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.bounds_ = {}

        for col in X_df.columns:
            q1 = X_df[col].quantile(0.25)
            q3 = X_df[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)

        for col, (low, high) in self.bounds_.items():
            X_df[col] = X_df[col].clip(lower=low, upper=high)

        return X_df.values


# =====================================
# Skewness Fixer
# =====================================
class SafeSkewFixer(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold
        self.transform_map_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.transform_map_ = {}

        for col in X_df.columns:
            sk = X_df[col].skew()

            if sk > self.skew_threshold:
                self.transform_map_[col] = "log"
            elif 0.5 < sk <= self.skew_threshold:
                self.transform_map_[col] = "sqrt"
            else:
                self.transform_map_[col] = "none"

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)

        for col in X_df.columns:
            action = self.transform_map_.get(col, "none")

            if action == "log":
                X_df[col] = np.log1p(X_df[col])
            elif action == "sqrt":
                X_df[col] = np.sqrt(X_df[col])

        return X_df.values

def load_model( model_file_name):
    """Load existing machine learning model"""
    if os.path.exists(model_file_name):
        try:
            with open(model_file_name, 'rb') as f:
                print ("{model_file_name}file exists")
                return joblib.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    else:
        print("{model_file_name} file does not exist")
        return None

def website_form_to_raw_df(form: dict) -> pd.DataFrame:
    """
    Expected keys (youthdiabetes.ai/risk groups 1–4):

      Group 1: age, gender, race, family_history
      Group 2: weight_lb, height_in, hypertension, high_cholesterol, pa_hours_week, screen_hours_day
      Group 3: protein_oz, dairy_cup, whole_grain_oz, fruit_cup, veg_cup
      Group 4: health_condition, food_stamps, eggs, poultry_oz, nuts_oz, milk_30days,
               solid_fat_g, oil_g, citrus_cup
    """

    # ----------------------------
    # Group 1
    # ----------------------------
    age = float(form["age"])
    RIAGENDR = map_gender_to_RIAGENDR(form["gender"])
    RaceEth = map_race_to_RaceEth(form["race"])
    fam_hist = yesno_to_01(form.get("family_history", "No"))

    # ----------------------------
    # Group 2
    # ----------------------------
    weight_lb = midpoint_from_range(form["weight_lb"])
    height_in = midpoint_from_range(form["height_in"])

    BMXWT = lb_to_kg(weight_lb) if not np.isnan(weight_lb) else np.nan
    BMXHT = inch_to_cm(height_in) if not np.isnan(height_in) else np.nan

    PAminWk = midpoint_from_range(form["pa_hours_week"]) * 60.0
    ScreenTime = float(form["screen_hours_day"])

    hypten = yesno_to_01(form["hypertension"])
    HTChol = yesno_to_01(form["high_cholesterol"])

    # ----------------------------
    # Group 3
    # ----------------------------
    PTot = midpoint_from_range(form["protein_oz"])
    DTot = midpoint_from_range(form["dairy_cup"])
    GWhole = midpoint_from_range(form["whole_grain_oz"])
    FTot = midpoint_from_range(form["fruit_cup"])
    VTot = midpoint_from_range(form["veg_cup"])

    # ----------------------------
    # Group 4 (mapped to your training raw columns)
    # ----------------------------
    # Note: keep text as-is for categorical fields; your preprocessor should handle encoding.
    HealthCond = form.get("health_condition", None)      # -> HealthCond (ordinal)
    FdStmp = form.get("food_stamps", None)               # Yes/No -> FdStmp (binary)

    PEgg =  midpoint_from_range(form["eggs"])             # -> PEgg (range)
    PPoult = midpoint_from_range(form["poultry_oz"])      # -> PPoult (range)
    PNut = midpoint_from_range(form["nuts_oz"])           # -> PNut (range)

    Milk30Days = form.get("milk_30days", None)           # -> Milk30Days (categorical)
    SolidFat = midpoint_from_range(form["solid_fat_g"])   # -> SolidFat (numeric or range)
    Oils = midpoint_from_range(form["oil_g"])             # -> Oils (numeric or range)

    # Closest match in your columns for "citrus cup"
    FCitMlB = midpoint_from_range(form["citrus_cup"])     # -> FCitMlB (numeric or range)

    raw_row = {
        # Group 1
        "RIDAGEYR": age,
        "RIAGENDR": RIAGENDR,
        "RaceEth": RaceEth,

        # Group 2
        "BMXWT": BMXWT,
        "BMXHT": BMXHT,
        "hypten": hypten,
        "HTChol": HTChol,
        "PAminWk": PAminWk,
        "ScreenTime": ScreenTime,

        # Group 3
        "PTot": PTot,
        "DTot": DTot,
        "GWhole": GWhole,
        "FTot": FTot,
        "VTot": VTot,

        # Group 4 mapped to training columns
        "HealthCond": HealthCond,
        "FdStmp": yesno_to_01(FdStmp) if FdStmp is not None else np.nan,

        "PEgg": midpoint_from_range(PEgg) if PEgg is not None else np.nan,
        "PPoult": midpoint_from_range(PPoult) if PPoult is not None else np.nan,
        "PNut": midpoint_from_range(PNut) if PNut is not None else np.nan,

        "Milk30Days": Milk30Days,
        "SolidFat": midpoint_from_range(SolidFat) if SolidFat is not None else np.nan,
        "Oils": midpoint_from_range(Oils) if Oils is not None else np.nan,

        "FCitMlB": midpoint_from_range(FCitMlB) if FCitMlB is not None else np.nan,
    }

    return pd.DataFrame([raw_row])

import sys
# Your class definitions here
sys.modules['__main__'].IQRClipper = IQRClipper
sys.modules['__main__'].SafeSkewFixer = SafeSkewFixer
# Then load your model

model1 = load_model('/var/data/youthdiabetes_logisticL1_scoring_bundle.pkl')

def predict(payload_4groups, prob_threshold=0.5):
    """Load existing machine learning model"""
    print("In predict function")
    print ('type of model:', type(model1))
    print('actual model1 keys:', model1.keys() if model1 else 'model1 is None')

    print('-------------------------------')
    print('model1 contents:', model1)
   
    preprocessor = model1["preprocessor"]
    raw_input_columns = model1["raw_input_columns"]
    post_feature_names = model1["post_feature_names"]
    selected_features = model1["selected_features"]
    default_threshold = float(model1.get("default_threshold", 0.52))
    thr = default_threshold if prob_threshold is None else float(prob_threshold)
    print("Model and preprocessor loaded successfully")

    payload_flat = {
        **payload_4groups["group1"],
        **payload_4groups["group2"],
        **payload_4groups["group3"],

        # group4 is optional; include it only if your model/preprocessor uses it
        **payload_4groups["group4"],
    }

    # ---------- Build raw row from website form ----------
    # Reuse your existing converter
    raw_df = website_form_to_raw_df(payload_flat)

    # Align to expected raw columns (critical)
    raw_aligned = raw_df.reindex(columns=raw_input_columns)

    # ---------- Transform ----------
    X_prep = preprocessor.transform(raw_aligned)

    if X_prep.shape[1] != len(post_feature_names):
        raise ValueError(
            f"Preprocessor output width mismatch: got {X_prep.shape[1]} cols, "
            f"expected {len(post_feature_names)}. Re-save bundle with correct post_feature_names."
        )

    X_prep_df = pd.DataFrame(X_prep, columns=post_feature_names, index=raw_aligned.index)


    # Sanity check: selected feature names must be a subset of the preprocessor output.
    missing_selected = [c for c in selected_features if c not in X_prep_df.columns]
    if missing_selected:
        raise ValueError(
        "Bundle mismatch: some `selected_features` are not present in the preprocessor output. "
        "This usually happens if the bundle was saved from a different run (or `post_feature_names` "
        "was computed differently). Missing examples: "
        f"{missing_selected[:20]} (missing_count={len(missing_selected)}). "
        "Fix: re-save the scoring bundle using `raw_input_columns=list(X_train_nontree.columns)`, "
        "`post_feature_names=list(X_train_non_tree_df.columns)`, and "
        "`selected_features=list(X_train_non_tree_sel.columns)` from the SAME training run."
        )

    # ---------- Select model inputs ----------
    X_sel = X_prep_df.loc[:, selected_features].copy()

    from sklearn.ensemble import RandomForestClassifier
    # ---------- Predict ----------
    proba = float(model1['model'].predict_proba(X_sel)[:, 1][0])
    pred = int(proba >= thr)

    # ---------- Build a compact summary ----------
    summary = pd.DataFrame([{
        "threshold": thr,
        "predicted_probability": proba,
        "predicted_class": pred
    }])
    print(summary.to_dict(orient='records')[0])
    print("Prediction completed successfully")
    return proba, pred, X_sel
    
@app.route('/api/evaluate', methods=['POST'])
def evaluate_risk():
    """Handle JSON POST data from frontend and evaluate risk"""
    print ('/evaluate_risk endpoint called')
    try:
        # Check if request contains JSON data
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        # Get JSON data from request
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400
        
        # Create report entry with metadata
        evaluation_data = {
            'timestamp': datetime.now().isoformat(),
            **json_data  # Include all data from frontend
        }

        print("Received evaluation data:", evaluation_data)  

        risk_factors = []

        ##############################################
        #find risk factors
        #calculate BMI
        weight = float(evaluation_data['weight'])
        height = float(evaluation_data['height'])
        bmi = weight / (height ** 2) * 703 if height > 0 else 0
        if bmi < 18.5:
            #risk_factors.append('Underweight')
            print ("BMI indicates Underweight")
        elif bmi >= 25:
            risk_factors.append('Overweight, High BMI')

        if evaluation_data['Hypertension'] == '1':
            hypertension_value = 1
            risk_factors.append('Hypertension')
            print("Hypertension value set to 1")
        else:   
            hypertension_value = 0
            print("Hypertension value set to 0")

        if evaluation_data['Cholesterol'] == '1':
            cholesterol_value = 1
            risk_factors.append('High Cholesterol')
            print("Cholesterol value set to 1")     
        else:   
            cholesterol_value = 0
            print("Cholesterol value set to 0")
        
        ###################################################
        # if factor is lower than median level, add to risk factors
        #convert Physical_Activity to float for comparison
        evaluation_data['Physical_Activity'] = float(evaluation_data['Physical_Activity'])
        if evaluation_data['Physical_Activity'] < 3.5:
            physical_activity_value = 1
            risk_factors.append('Low Physical Activity')
            print("Physical Activity value set to 1")
        else:   
            physical_activity_value = 0
            print("Physical Activity value set to 0")

        #convert Screen_Time to float for comparison
        evaluation_data['Screen_Time'] = float(evaluation_data['Screen_Time'])  
        if evaluation_data['Screen_Time'] > 5:
            screen_time_value = 1
            risk_factors.append('High Screen Time')
            print("Screen Time value set to 1")
        else:   
            screen_time_value = 0
            print("Screen Time value set to 0")

        #convert dietary intake to float for comparison
        evaluation_data['protein'] = float(evaluation_data['protein'])
        if evaluation_data['protein'] < 5.29:
            protein_value = 1
            risk_factors.append('Low Protein Intake')
            print("Protein Intake value set to 1") 
        else:   
            protein_value = 0
            print("Protein Intake value set to 0")

        #convert dietary intake to float for comparison
        evaluation_data['dairy'] = float(evaluation_data['dairy'])
        if evaluation_data['dairy'] < 1.5:
            dairy_value = 1
            risk_factors.append('Low Dairy Intake')
            print("Dairy Intake value set to 1")
        else:   
            dairy_value = 0
            print("Dairy Intake value set to 0")

        #convert grain intake to float for comparison
        evaluation_data['grain'] = float(evaluation_data['grain'])
        if evaluation_data['grain'] < 6.5:
            grain_value = 1
            risk_factors.append('Low Whole Grain Intake')
            print("Grain Intake value set to 1")
        else:   
            grain_value = 0
            print("Grain Intake value set to 0")

        #convert fruit intake to float for comparison
        evaluation_data['fruit'] = float(evaluation_data['fruit'])
        if evaluation_data['fruit'] < 0.4:
            fruit_value = 1
            risk_factors.append('Low Fruit Intake')
            print("Fruit Intake value set to 1")
        elif evaluation_data['fruit'] > 6.0:
            fruit_value = 1
            risk_factors.append('High Fruit Intake, Monitor Sugar Level')
            print("Fruit Intake value set to 1")    
        else:   
            fruit_value = 0
            print("Fruit Intake value set to 0")

        #convert vegetable intake to float for comparison
        evaluation_data['vegetable'] = float(evaluation_data['vegetable'])
        if evaluation_data['vegetable'] < 0.88:
            vegetable_value = 1
            risk_factors.append('Low Vegetable Intake')
            print("Vegetable Intake value set to 1")
        else:   
            vegetable_value = 0
            print("Vegetable Intake value set to 0")
        
        # For testing purposes, simulate prediction
        """payload = {
            "age": 16,
            "gender": "Male",
            "race": "Non Hispanic White",
            "family_history_diabetes": "Yes",
            "weight_lb": "151-170",
            "height_in": "66-75",
            "hypertension": "No",
            "high_cholesterol": "No",
            "pa_hours_week": "3",
            "screen_hours_day": "5",
            "protein_oz": "7-8",
            "dairy_cup": "1",
            "whole_grain_oz": "1-2",
            "fruit_cup": "0.6-0.75",
            "veg_cup": "0.75-1",
        }"""
    
        # Example website payload with 4 groups (nested, website-style)
        payload_4groups = {
        "group1": {  # Group 1 Factors
            "age": evaluation_data['age'],
            "gender": evaluation_data['gender'],
            "race": evaluation_data['race'],
            "family_history": evaluation_data['family_history']
        },
        "group2": {  # Group 2 Factors
            # "weight_lb": "151-170",
            "weight_lb": evaluation_data['weight'],
            "height_in": evaluation_data['height'],
            "hypertension": evaluation_data['Hypertension'],
            "high_cholesterol": evaluation_data['Cholesterol'],
            "pa_hours_week": evaluation_data['Physical_Activity'],
            "screen_hours_day": evaluation_data['Screen_Time']
        },
        "group3": {  # Group 3 Factors
            "protein_oz": evaluation_data['protein'],
            "dairy_cup": evaluation_data['dairy'],
            "whole_grain_oz": evaluation_data['grain'],
            "fruit_cup": evaluation_data['fruit'],
            "veg_cup": evaluation_data['vegetable']
        },
        #field might not have value, so make it optional
        "group4": {  # Group 4 Factors (optional, advanced)
            "health_condition": evaluation_data['health_condition'],
            "food_stamps": evaluation_data['food_stamps'],
            "eggs": evaluation_data['eggs'],
            "poultry_oz": evaluation_data['poultry'],
            "nuts_oz": evaluation_data['nuts'],
            "milk_30days": evaluation_data['milk'],
            "solid_fat_g": evaluation_data['solid_fat'],
            "oil_g": evaluation_data['oil'],
            "citrus_cup": evaluation_data['citrus']
        }
        }

        probability, prediction, features = predict(payload_4groups, prob_threshold=0.5)
        
        evaluation_result = {
            'probability': int(probability*100),
            'risk_level': 'High' if prediction == 1 else 'Low',
            'risk_factors': risk_factors,
        }

        print("Evaluation result:", evaluation_result)  

        return jsonify({
            'success': True,
            'message': 'Report submitted successfully',
            'evaluation_result': evaluation_result
        }), 201
        
    except json.JSONDecodeError:
        return jsonify({
            'success': False,
            'error': 'Invalid JSON format'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json

    print("Received data for ChatGPT:", data)
    user_message = data.get("prompt", "")

    print("Received message for ChatGPT:", user_message)

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Call the ChatGPT API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Or another model like gpt-4, gpt-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1500
        )

        ai_text = response.choices[0].message.content.strip()
        print ('ai_text:', ai_text)
        
        return jsonify({"reply": ai_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create JSON files if they don't exist
    app.run(debug=True, host='0.0.0.0', port=10000)
