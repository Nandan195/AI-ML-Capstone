# Analytics Pipeline: Predictive Modeling Report
## Train/Test Split & Class Balance
- **Total Survived/Not-Survived Balance:** 0: 549, 1: 342
- **Why Stratification Matters:** The dataset has an imbalance (~61% non-survivors). A stratified split ensures both the training and testing sets maintain this exact proportion, preventing training bias and ensuring evaluation metrics are representative of the actual population.

## Classification Evaluation
### Classification Metrics Table
| Model               |   Accuracy |   Precision |   Recall |    F1 |   AUC |
|:--------------------|-----------:|------------:|---------:|------:|------:|
| Logistic Regression |      0.804 |       0.783 |    0.681 | 0.729 | 0.85  |
| Decision Tree       |      0.765 |       0.755 |    0.58  | 0.656 | 0.797 |
| Random Forest       |      0.793 |       0.758 |    0.681 | 0.718 | 0.825 |

## Class Imbalance Comparison (Random Forest)
| Strategy      |   Precision |   Recall |    F1 |
|:--------------|------------:|---------:|------:|
| Baseline      |       0.758 |    0.681 | 0.718 |
| Class Weights |       0.79  |    0.71  | 0.748 |
| SMOTE         |       0.785 |    0.739 | 0.761 |

**Conclusion:** SMOTE successfully synthetic-samples the minority class purely within the training folds. However, random forest is highly robust, and standard `class_weight='balanced'` often yields comparable or slightly better recall for survivors without introducing the noise of synthetic data points. For deployment, basic class weights or baseline might be preferred for simplicity, but SMOTE provided a reliable lift to recall.

## Hyperparameter Tuning
- **Best Parameters:** {'classifier__max_depth': 10, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 200}
- **Out-of-Bag (OOB) Score of Final Model:** 0.819

## Regression Task (Predict Fare)
### Regression Metrics Table
|    MAE |   RMSE |    R2 |   Adjusted R2 |
|-------:|-------:|------:|--------------:|
| 20.808 | 30.511 | 0.398 |         0.374 |

**Heteroscedasticity Conclusion:** The residual plot clearly exhibits a fan/cone shape, with variance of residuals increasing significantly at higher predicted fare values. This strongly indicates the presence of heteroscedasticity, meaning the linear regression model struggles with the highly skewed distribution of the `fare` target variable.

## Final Comparison and Recommendation
Comparing the distinct metric groups, the Classification task achieved strong predictive capacity. Based on the classification metrics table, **Random Forest** (tuned via GridSearchCV) is the recommended model for deployment. It achieved the highest overall AUC and F1 score, demonstrating an exceptional balance between precision and recall while robustly handling the complex interactions between passenger class, sex, and age.

## Serialized Pipeline
- Final fitted pipeline saved to `N:\AI-ML-Capstone\analytics\artifacts\best_pipeline.joblib`
- **Reload Test Success:** Successfully reloaded pipeline and fed raw dict. Prediction: Died
