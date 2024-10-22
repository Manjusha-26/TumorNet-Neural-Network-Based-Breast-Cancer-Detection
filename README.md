# TumorNet-Neural-Network-Based-Breast-Cancer-Detection

## Project Overview
Breast cancer remains one of the most prevalent health challenges globally, affecting millions of women each year. Early detection of breast cancer is crucial for improving patient outcomes and reducing the costs associated with late-stage treatments. This project focuses on developing a **machine learning model** using **XGBoost** to classify breast cancer tumors as either **malignant** (cancerous) or **benign** (non-cancerous), providing healthcare professionals with data-driven tools for timely and accurate diagnosis.

## Business Problem
The early detection and classification of breast cancer tumors is vital for healthcare providers. Incorrect classification of a tumor can result in delayed care or unnecessary treatments, leading to poor patient outcomes and increased healthcare costs. Deploying machine learning models in medical diagnostics can improve both the accuracy and efficiency of tumor classification.

### Key Business Challenges
- **Accurate Diagnosis**: Misclassification of tumors can lead to delayed treatment or unnecessary interventions. A reliable machine learning model helps improve diagnostic accuracy.
- **Cost Management**: Malignant tumors often require costly, long-term treatments. Early detection helps manage healthcare costs by preventing advanced-stage cancer treatments.
- **Healthcare Efficiency**: Implementing machine learning models in healthcare can reduce the time and effort required for manual tumor diagnosis, making the system more efficient and scalable.

## Objective
The objective of this project is to develop a machine learning model using **XGBoost** that accurately classifies breast cancer tumors as malignant or benign based on diagnostic features extracted from medical data. This model will help healthcare providers make data-driven, timely, and reliable diagnoses, improving patient care and managing the costs associated with cancer treatment.

## Approach

1. **Data Preprocessing**:
   - Feature scaling was performed to normalize the input data.
   - Categorical variables were one-hot encoded for use in the model.
   - Missing data handling and removal of irrelevant features was completed.

2. **Exploratory Data Analysis (EDA)**:
   - The dataset was explored to understand the distribution of malignant vs. benign tumors.
   - Correlation analysis between features was conducted to identify the most significant features.

3. **Feature Selection**:
   - Diagnostic features, such as radius, texture, smoothness, compactness, and symmetry of the tumor, were considered.
   - XGBoost was used to compute feature importance.

4. **Model Building**:
   - **XGBoost** was used for tumor classification. The model was trained on labeled medical data to distinguish between malignant and benign tumors.
   - Hyperparameter tuning was applied to optimize the model for better performance.

5. **Model Evaluation**:
   - Accuracy, precision, recall, F1-score, and AUC-ROC curve were used to evaluate model performance.
   - Cross-validation was performed to assess model robustness and prevent overfitting.

## Results
- The **XGBoost** model achieved a high level of accuracy in classifying breast cancer tumors.
- Feature importance analysis revealed the most influential diagnostic features in the prediction process.

## Tools and Technologies
- **Python**: Used for model building, data preprocessing, and evaluation.
- **XGBoost**: The main algorithm used for tumor classification.
- **Pandas, NumPy, Scikit-learn**: Libraries for data manipulation, analysis, and preprocessing.
- **Matplotlib, Seaborn**: For visualizing data and results.

## Conclusion
The machine learning model developed in this project provides an efficient and accurate method to classify breast cancer tumors as malignant or benign. This tool can assist healthcare providers in making data-driven decisions, ultimately improving patient outcomes and reducing healthcare costs.

## Future Work
- Experiment with other advanced machine learning algorithms such as **Neural Networks** and **Random Forest** for potential improvement.
- Explore additional diagnostic features or external datasets to improve the model's robustness.
- Implement real-time deployment in a clinical setting for testing and feedback.
