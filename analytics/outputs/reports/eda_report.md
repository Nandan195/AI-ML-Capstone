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
