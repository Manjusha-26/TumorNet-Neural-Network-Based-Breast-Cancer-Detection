# TumorNet: Neural Network-Based Breast Cancer Detection

## Project Overview
Breast cancer is one of the most critical health challenges worldwide, affecting millions of women every year. Early and accurate detection is crucial for improving patient outcomes and managing healthcare costs. This project focuses on developing multiple **machine learning models**, including **Neural Networks**, to classify breast cancer tumors as either **malignant** (cancerous) or **benign** (non-cancerous). The goal is to evaluate each model's performance and select the most accurate one, providing healthcare providers with a reliable diagnostic tool.

## Business Problem
Early and accurate detection of breast cancer can significantly improve patient outcomes and reduce the costs of treatment, especially in the advanced stages of cancer. Misclassification of a tumor could lead to delayed care or unnecessary treatments, which could increase healthcare expenses and decrease the patient's quality of life. Implementing machine learning models into medical diagnostics offers the potential to automate and enhance the accuracy of breast cancer classification.

### Key Business Challenges
- **Accurate Diagnosis**: Reducing the chances of misclassification and ensuring timely treatment.
- **Cost Management**: Early detection and treatment of malignant tumors help to lower long-term healthcare costs.
- **Healthcare Efficiency**: Using machine learning models to automate diagnostics reduces manual efforts and improves healthcare workflows.

## Objective
The objective is to build and evaluate multiple machine learning models to classify breast cancer tumors as malignant or benign. After training, each model is evaluated to identify the most accurate one, ultimately providing healthcare providers with a data-driven, reliable tool for timely and accurate tumor classification.

## Approach

1. **Data Preprocessing**:
   - Feature scaling to normalize the input data.
   - Converted categorical variables like diagnosis to numerical values (malignant = 1, benign = 0).
   - Checked and handled missing data, and removed irrelevant features.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed the dataset to understand the distribution of malignant and benign tumors.
   - Generated correlation heatmaps and pair plots to understand relationships between features.

3. **Feature Selection**:
   - Key features such as radius, texture, smoothness, compactness, and symmetry were used for tumor classification.
   - Feature selection was based on correlation analysis and domain knowledge.

4. **Model Building**:
   - Developed and compared multiple models, including a **Neural Network** and traditional algorithms such as **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, and **Naive Bayes**.
   - Each model was tuned for optimal performance using hyperparameter optimization.

5. **Model Evaluation and Selection**:
   - Evaluated all models based on metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **AUC-ROC**.
   - Cross-validation and confusion matrices were used to assess each model’s robustness and generalizability.
   - The model with the highest performance across all metrics was selected as the final model.

## Results
- The **Neural Network** model achieved high accuracy, precision, and recall, making it a strong candidate for final deployment.
- Analysis highlighted the most influential diagnostic features, ensuring the model relied on relevant data for predictions.

## Tools and Technologies
- **Python**: Used for model building, data preprocessing, and evaluation.
- **TensorFlow**: Used for developing and training the Neural Network.
- **Pandas, NumPy**: For data manipulation and analysis.
- **Matplotlib, Seaborn**: For visualizing data distributions and model performance.

## Conclusion
The project successfully developed multiple models to classify breast cancer tumors. The selected model provides an efficient and accurate method for distinguishing between malignant and benign tumors, enabling healthcare providers to make data-driven decisions, ultimately improving patient outcomes and reducing healthcare costs.

## Future Development
- Create a front-end application for clinical use, allowing healthcare providers to input patient data and receive real-time predictions from the selected model.
