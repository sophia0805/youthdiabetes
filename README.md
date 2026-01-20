# youthdiabetes.ai | AI-Powered Youth Diabetes Risk Assessment

an AI-powered web application for assessing youth diabetes risk!
> built to help teens and families evaluate diabetes risk using machine learning models trained on Mount Sinai School of Medicine data and receive personalized AI-generated recommendations

## features
- **Homepage** with youth diabetes crisis statistics and animated prediabetes counter showing 8.4 million teens at risk
- **Risk Assessment Tool** (`/risk`) with comprehensive health questionnaire covering:
  - Demographics (age, gender, race, family history)
  - Physical factors (weight, height, BMI, hypertension, cholesterol)
  - Lifestyle factors (physical activity, screen time)
  - 24-hour dietary intake (protein, dairy, whole grains, fruits, vegetables)
- **Machine Learning Predictions** using logistic regression models trained on the most recent comprehensive youth diabetes dataset (Mount Sinai School of Medicine, 2024) covering 15,149 individuals aged 12-19
- **AI-Generated Recommendations** powered by OpenAI GPT-4o-mini for personalized daily meal plans, weekly exercise schedules, and lifestyle modifications
- **Resources Page** with curated links to diabetes education, research organizations, and support groups
- **About Page** explaining the technical approach, responsible AI practices, and model validation methods

## installation

1. **clone the repository**
   ```bash
   git clone https://github.com/sophia0805/youthdiabetes-ai
   cd youthdiabetes-ai
   ```

2. **set up Python virtual environment**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **create a `.env` file in the backend directory**
   ```.env
   OPENAI_API_KEY="your_openai_api_key"
   ```

5. **place model files** (if you have them)
   - Ensure `.pkl` model files are in the `backend/` directory:
     - `youthdiabetes_logisticL1_scoring_bundle.pkl`
     - `youthdiabetes_scoring_bundle.pkl`
     - `youthdiabetes_scoring_bundle14.pkl`

6. **run the development server**
   ```bash
   python3 app.py
   ```

7. **open your browser**
   navigate to [http://127.0.0.1:10000](http://127.0.0.1:10000)

## deployment

For production deployment on Render Cloud Application Platform:

**Start Command:**
```bash
gunicorn app:app --chdir backend --bind 0.0.0.0:$PORT
```

**Environment Variables:**
- `OPENAI_API_KEY`: Your OpenAI API key for GPT-4o-mini generative AI recommendations

**Note:** Model files are large (several MB each) and may need special handling for deployment platforms with size limits.

## technical details

### Machine Learning Model
- **Algorithm:** Logistic Regression
- **Performance:** Weighted ROC-AUC of 0.68, F1 Score of 0.71
- **Training Data:** Mount Sinai School of Medicine youth diabetes dataset (2024)
  - Source: National Health and Nutrition Examination Survey (NHANES) 1999-2018
  - Sample Size: 15,149 individuals aged 12-19
  - Features: ~30-40 key predictors selected from 100+ variables through hybrid feature selection
  - Target: Prediabetes status (preDM2) with ~13% prevalence

### Generative AI
- **Model:** OpenAI GPT-4o-mini
- **Validation:** LLM-as-a-Judge method using multiple AI systems (ChatGPT, Deepseek, Gemini, Grok) for cross-validation
- **Purpose:** Generate personalized daily meal plans, weekly exercise schedules, and lifestyle recommendations

### System Architecture
- **Frontend:** HTML, CSS, JavaScript (Vanilla)
- **Backend:** Flask (Python)
- **ML Libraries:** scikit-learn, pandas, numpy, joblib
- **AI Integration:** OpenAI API
- **Deployment:** Render Cloud Application Platform

## history
- identified and acquired the most recent comprehensive youth diabetes dataset from Mount Sinai School of Medicine (released July 2024)
- trained multiple machine learning models (XGBoost, AdaBoost, Random Forest, Logistic Regression, MLP, Ensemble models)
- implemented hybrid feature selection pipeline reducing 100+ variables to 30-40 key predictors
- selected logistic regression for deployment based on balance of interpretability, computational efficiency, and predictive performance
- built comprehensive web application integrating ML predictions with generative AI recommendations
- struggled with slow response times initially due to training models on-demand, solved by deploying pre-trained serialized pickle models
- integrated OpenAI API for personalized health plan generation
- validated generative AI outputs using LLM-as-a-Judge cross-validation method across multiple AI systems
- fixed Flask static file serving and template rendering issues during reorganization
- deployed to Render Cloud Application Platform for public access

## community impact
This application creates significant positive impact in communities with limited healthcare resources by enabling early diabetes risk assessment through machine learning and providing personalized risk management plans using generative AI. In underserved and rural areas where access to medical professionals is limited, the application offers an affordable and accessible way for individuals to understand their diabetes risk without frequent clinic visits.

The tool helps users adopt healthier habits through actionable daily meal suggestions, specific exercise routines tailored to teenagers, and practical lifestyle changes. This is especially beneficial for individuals with low health literacy, as the guidance is easy to understand and available at any time.

## future enhancements
- retrain machine learning models with newer youth diabetes datasets as they become available
- evaluate alternative generative AI solutions from various providers to identify optimal platforms
- incorporate additional health indicators and risk factors as research advances
- improve model accuracy and expand feature set based on emerging research

## references
- [National Diabetes Statistics Report](https://www.cdc.gov/diabetes/data/statistics-report/index.html) - CDC
- [Youth preDM/DM Dataset](https://zenodo.org/records/8206576) - Mount Sinai School of Medicine
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/introduction)
- [Render Cloud Platform](https://render.com/docs/web-services)

## live website
🌐 [www.youthdiabetes.ai](https://www.youthdiabetes.ai)
