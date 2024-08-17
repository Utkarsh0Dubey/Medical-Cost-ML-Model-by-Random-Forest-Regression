# Medical Cost Estimation: Random Forest

## Project Overview
This project aims to predict medical insurance costs based on features such as age, BMI, smoking status, and region. The dataset contains information about individuals and their insurance costs, with both categorical and numerical features. A Random Forest Regressor is used to improve predictions over a simpler linear regression model by capturing non-linear relationships and feature interactions.

## Dataset
- **Source**: The dataset used in this project is a medical insurance dataset, commonly available on platforms like Kaggle and other machine learning repositories.
- **Features**: 
  - `age`: Age of the individual
  - `sex`: Gender of the individual
  - `bmi`: Body Mass Index
  - `children`: Number of children/dependents
  - `smoker`: Whether the individual is a smoker
  - `region`: Region where the individual resides
- **Target**: 
  - `charges`: Medical insurance costs for the individual

## Project Steps
1. **Data Preprocessing**: 
   - Encoding categorical variables (e.g., `sex`, `smoker`, `region`) using one-hot encoding.
   - Feature scaling to standardize the numerical features before training the model.
2. **Model Training**:
   - Used `scikit-learn`'s `RandomForestRegressor` with 100 trees (`n_estimators=100`) to train the model on 80% of the data (training set).
3. **Model Evaluation**:
   - Evaluated the model on both training and testing sets using Mean Squared Error (MSE) and R-squared metrics.
   - Generated visualizations to compare actual vs. predicted values for both training and testing data.

## Results
- **Training MSE**: Approx. 5000
- **Testing MSE**: Approx. 2000
- **Conclusion**: The Random Forest model significantly improved predictions compared to linear regression, particularly on the testing set. The model captures non-linear relationships and interactions between features, making it more robust.

## How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Utkarsh0Dubey/Medical-Cost-ML-Model-by-Random-Forest-Regression.git
   cd Medical-Cost-ML-Model-by-Random-Forest-Regression
2. **Getting missing requirements**
pip install -r requirements.txt
