
# Advanced Sales Prediction Project

This project uses an optimized XGBoost model to predict sales revenue based on advertising spend on TV, Radio, and Newspaper. It includes a full data analysis pipeline and an interactive web application built with Streamlit.

## Project Structure

- `data/`: Contains the raw `Advertising.csv` dataset.
- `model/`: Stores the trained XGBoost model and feature names.
- `notebooks/`: Includes the Jupyter Notebook with the full analysis and model training process.
- `app.py`: The Streamlit web application for live predictions.
- `requirements.txt`: A list of required Python libraries.

## How to Run

1.  **Clone the repository.**
2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optional) Explore the analysis:**
    Open and run the `notebooks/1_Sales_Prediction_Model_Training.ipynb` file in Jupyter Lab/Notebook to see how the model was built.
4.  **Launch the Web App:**
    Run the following command in your terminal from the project's root directory:
    ```bash
    streamlit run app.py
    ```
=======
# Advanced-Sales-Prediction-with-XGBoost-and-Streamlit

## Project Overview

This project demonstrates an end-to-end data science workflow for predicting product sales based on advertising spend. It utilizes the classic "Advertising.csv" dataset but elevates the analysis through advanced techniques.

The project covers:
-   **Exploratory Data Analysis (EDA):** To understand the relationships between advertising channels and sales.
-   **Advanced Feature Engineering:** Creating new features (interaction terms, budget shares) to capture complex patterns.
-   **Predictive Modeling:** Using a powerful, optimized XGBoost Regressor.
-   **Hyperparameter Tuning:** Employing GridSearchCV to find the best model configuration.
-   **Interactive Web Application:** A user-friendly web app built with Streamlit to deliver real-time predictions.

---

## Model Performance

The final, tuned XGBoost model achieved outstanding performance on the test set:

-   **R-squared (R²) Score:** [Your Final R² Score, e.g., 0.9851]
-   **Mean Absolute Error (MAE):** [Your Final MAE, e.g., 0.55]

An R² score of [Your R² Score] indicates that the model can explain approximately **[Your R² Score * 100]%** of the variance in the sales data.

---

## Key Findings & Feature Importance

The model's feature importance analysis revealed that:
- The engineered feature `TV_Radio_Interaction` is a highly significant predictor, confirming the synergistic effect of these two channels.
- `Radio` and `TV` spend remain the primary drivers of sales.
- The original `Newspaper` feature was dropped due to its low predictive power, a key modeling decision.

![Feature Importance Chart](path/to/your/feature_importance_chart.png)  
*(Optional: Take a screenshot of your feature importance plot and add it to an `images` folder to display it here.)*

>>>>>>> 855eb99834db167068737182eb746a1aa1aae412
## 🎥 Project Demo

Watch a full execution of the Streamlit app in action:

[Click here to watch the project execution video](media/project_execution.mp4)

