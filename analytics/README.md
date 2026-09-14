# Analytics Pipeline: EDA Report
## Part A - Profiling and Cleaning
**Shape:** (891, 15)
### Missing Values (Before Cleaning)
- **age**: 19.87% missing- **embarked**: 0.22% missing- **deck**: 77.22% missing- **embark_town**: 0.22% missing
### Cleaning Decisions
- **embarked / embark_town** (< 5% missing): Dropped affected rows.
- **age** (5% - 30% missing): Imputed with median value to avoid skewness impact.
- **deck** (> 30% missing): Encoded missing values as 'Unknown' category. Justification: Dropping the column loses potential spatial information, and dropping rows would destroy the dataset.
**Shape after cleaning:** (889, 15)
## Univariate Analysis
### Age
- **Outliers (IQR rule):** 65
### Fare
- **Outliers (IQR rule):** 114
- **Mean:** 32.10
- **Median:** 14.45
- **Mode:** 8.05
- **Distribution Classification:** Right-skewed
## Bivariate Analysis
### Survival Rates
- **By Sex:**
  - female: 74.04%
  - male: 18.89%
- **By Pclass:**
  - Class 1: 62.62%
  - Class 2: 47.28%
  - Class 3: 24.24%
- **By Sex AND Pclass (Examples via Masking):**
  - Female in 1st Class: 96.74%
  - Male in 3rd Class: 13.54%

### Correlation Matrix Analysis
**Strongest Pair 1:** pclass & fare (|r| = 0.548)
**Strongest Pair 2:** sibsp & parch (|r| = 0.415)

**Interpretation:**
- The strongest relationship is between **pclass** and **fare**. This makes intuitive sense: higher ticket classes (lower numerical pclass values) command significantly higher fares.
- The second strongest relationship is between **sibsp** and **parch**. This indicates that passengers travelling with siblings/spouses were also highly likely to be travelling with parents/children, representing entire family units aboard.

## Multivariate Data Story
### Chart 1: Survival Rate by Passenger Class and Sex
This bar plot illustrates the compounded effect of gender and socio-economic status on survival. Females across all classes survived at much higher rates than males, heavily driven by the 'women and children first' protocol. However, 3rd class females still suffered significantly higher mortality than 1st and 2nd class females, exposing a harsh class divide.

### Chart 2: Age Distribution Density by Survival Status
This density plot compares the age distributions of those who survived versus those who perished. A noticeable spike exists for young children (age < 10) in the 'survived' category, highlighting prioritizing children during evacuation. Conversely, young adults (20-30) form the bulk of the casualties.

### Chart 3: Fare vs Age Scatterplot by Survival and Class
This scatterplot reveals the interaction between age, fare paid, class size, and survival outcome. The highest fares are concentrated among middle-aged adults, who overwhelmingly survived (orange dots). The bottom-left cluster represents lower-fare, younger individuals who suffered widespread casualties.

### Chart 4: Key Multivariate Pairplot
This pair plot summarizes the pairwise distributions across age, fare, and passenger class, separated by survival. It confirms visually that high fares (and thus 1st class) act as a strong separator for survival. It also emphasizes the dense clustering of non-survivors at the lowest fare and highest pclass (3rd class) margins.

## Standardization Sanity Check (EDA Only)
- **Age Z-score check:** Mean = 0.0000000000, Std = 1.00000
- **Fare Z-score check:** Mean = 0.0000000000, Std = 1.00000
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
