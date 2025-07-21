Title:

AI-Powered Streamlit App for Predicting Ebola Case Reduction Using NPIs



Introduction

In recent years, outbreaks such as Ebola have posed major health challenges across the globe, particularly in low-resource regions. 
Predicting disease trends and identifying effective interventions can help governments and health agencies make informed decisions. 
This project aims to use machine learning techniques to forecast Ebola case reduction based on various non-pharmaceutical interventions (NPIs), addressing Sustainable Development Goal 3 (SDG 3): Good Health and Well-being.



Methodology

1. Data Sources

OWID (Our World in Data): Provided historical disease data, including Ebola case counts.

OxCGRT (Oxford COVID-19 Government Response Tracker): Contained detailed information on government policy responses (e.g., lockdowns, school closures, travel restrictions), repurposed for Ebola context.


2. Data Preparation

Cleaned and standardized date and country formats.

Merged datasets based on country and date.

Handled missing values using forward-fill and interpolation methods.


3. Feature Engineering

Selected key NPIs (e.g., stringency index, health containment index).

Created lag variables to model delayed effects of NPIs on case trends.


4. Model Development

Used machine learning models such as Random Forest and Linear Regression to learn patterns between NPIs and Ebola case trends.

Evaluated performance using Root Mean Squared Error (RMSE) and R² Score.


5. Application Interface

Built using Streamlit, the app allows users to input NPI values and get predictions on the expected reduction in Ebola cases.


Results

The model demonstrated a high correlation between strong health policies and case reduction.

Random Forest outperformed simpler models in capturing nonlinear interactions between NPIs and disease trends.

The app provides real-time visual feedback and prediction outputs in a user-friendly interface.


Conclusion

This project demonstrates how AI and open datasets can support global health initiatives. By modeling the impact of policy interventions on disease spread, the system supports SDG 3 by:

Promoting timely responses to outbreaks

Encouraging data-driven public health planning

Making disease forecasting accessible even in low-resource settings


Alignment with SDG 3 (Good Health and Well-being)

Target 3.3: Helps end epidemics by enhancing prediction of case trends.

Target 3.d: Strengthens early warning and risk reduction through predictive modeling.

Equity Focus: App designed to work with open data, ensuring access for low-income countries.

Ethical AI: The model emphasizes fairness, transparency, and interpretability.

