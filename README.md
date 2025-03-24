# 🏡 Belgian Real Estate Price Predictor

This is a Streamlit web application for predicting real estate prices in Belgium using a trained Random Forest Regression model. The data was sourced from [immoweb.be](https://immoweb.be), a popular real estate platform in Belgium.

Users can input various property features (like number of bedrooms, living area, equipped kitchen, etc.) and receive a predicted price instantly.

---

## 🚀 Getting Started

### 🔧 Installation

Make sure you have Python 3.9+ installed. Then, install dependencies using Poetry:

```bash
poetry install

▶️ Run the App

To start the Streamlit app locally:

streamlit run streamlit_app.py

The app will open in your browser at http://localhost:8501.

📁 Project Structure

.
├── .devcontainer/
│   └── devcontainer.json
├── .idea/
│   └── inspectionProfiles/
│       └── ...
│   ├── immo-eliza-deployment.iml
│   ├── misc.xml
│   ├── modules.xml
│   └── vcs.xml
├── models/
│   └── trained_rf_model.joblib     # Pretrained Random Forest Regression model
├── .gitignore
├── README.md
├── poetry.lock
├── pyproject.toml
└── streamlit_app.py                # Main Streamlit app

🧠 Model Info
Type: RandomForestRegressor from scikit-learn
	•	Serialized with: joblib
	•	Input Features:
	•	Number of bedrooms
	•	Living area (m²)
	•	Equipped kitchen (Yes/No)
	•	Furnished (Yes/No)
	•	Swimming pool (Yes/No)
	•	Building condition (6 levels)
	•	Property type (House/Apartment)
	•	One-hot encoded Belgian provinces

📬 Support

For questions or issues, please open an issue or contact the developer via GitHub.
