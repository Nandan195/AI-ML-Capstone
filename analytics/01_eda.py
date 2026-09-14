import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(os.path.dirname(__file__), 'titanic.csv')

def load_data():
    """Load dataset exactly ONCE, save to CSV, and return."""
    if not os.path.exists(CSV_PATH):
        print("Downloading titanic dataset from seaborn...")
        df = sns.load_dataset("titanic")
        df.to_csv(CSV_PATH, index=False)
    else:
        print("Loading titanic dataset from local CSV...")
        df = pd.read_csv(CSV_PATH)
    return df

def generate_eda_report(df):
    report_lines = []
    report_lines.append("# Analytics Pipeline: EDA Report\n")
    
    # 1. Profiling
    report_lines.append("## Part A - Profiling and Cleaning\n")
    report_lines.append(f"**Shape:** {df.shape}\n")
    
    # Missing percentages before cleaning
    report_lines.append("### Missing Values (Before Cleaning)\n")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col, pct in missing_pct[missing_pct > 0].items():
        report_lines.append(f"- **{col}**: {pct:.2f}% missing")
    report_lines.append("\n")
    
    # Cleaning decisions
    report_lines.append("### Cleaning Decisions\n")
    
    # Apply thresholds
    # <5% drop rows, 5-30% impute, >30% drop or encode
    
    # For titanic:
    # embarked, embark_town ~0.22% missing -> Drop rows
    # age ~19.8% missing -> Impute
    # deck ~77% missing -> Extremely high missingness. Decision: Encode as category 'Unknown'. Justification: Dropping 77% of rows would destroy the dataset, and dropping the column entirely might lose the signal that 'knowing the deck' implies higher class/survival.
    
    report_lines.append("- **embarked / embark_town** (< 5% missing): Dropped affected rows.\n")
    report_lines.append("- **age** (5% - 30% missing): Imputed with median value to avoid skewness impact.\n")
    report_lines.append("- **deck** (> 30% missing): Encoded missing values as 'Unknown' category. Justification: Dropping the column loses potential spatial information, and dropping rows would destroy the dataset.\n")
    
    # Perform cleaning
    if 'deck' in df.columns:
        if pd.api.types.is_categorical_dtype(df['deck']) or isinstance(df['deck'].dtype, pd.CategoricalDtype):
            df['deck'] = df['deck'].cat.add_categories('Unknown')
        df['deck'] = df['deck'].fillna('Unknown')
    
    # 2. age -> impute median
    age_median = df['age'].median()
    df['age'] = df['age'].fillna(age_median)
    
    # 3. embarked, embark_town -> drop rows
    df = df.dropna(subset=['embarked', 'embark_town'])
    
    report_lines.append(f"**Shape after cleaning:** {df.shape}\n")
    
    # 2. Univariate Analysis
    report_lines.append("## Univariate Analysis\n")
    
    for col in ['age', 'fare']:
        # Plot Histogram & Boxplot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Histogram of {col}')
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'Box Plot of {col}')
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'univariate_{col}.png')
        plt.savefig(plot_path)
        plt.close()
        
        # IQR outliers
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        report_lines.append(f"### {col.capitalize()}\n")
        report_lines.append(f"- **Outliers (IQR rule):** {len(outliers)}\n")
        
        if col == 'fare':
            mean_val = df[col].mean()
            median_val = df[col].median()
            mode_val = df[col].mode()[0]
            report_lines.append(f"- **Mean:** {mean_val:.2f}\n")
            report_lines.append(f"- **Median:** {median_val:.2f}\n")
            report_lines.append(f"- **Mode:** {mode_val:.2f}\n")
            
            skewness = "Right-skewed" if mean_val > median_val > mode_val or mean_val > median_val else "Left-skewed" if mean_val < median_val else "Symmetric"
            report_lines.append(f"- **Distribution Classification:** {skewness}\n")
            
    # 3. Bivariate Analysis
    report_lines.append("## Bivariate Analysis\n")
    
    # Survival Rates
    surv_sex = df.groupby('sex')['survived'].mean()
    surv_pclass = df.groupby('pclass')['survived'].mean()
    
    # sex AND pclass using boolean masking (as requested, though groupby is easier, we will compute explicitly)
    # Masking example:
    mask_female_1 = (df['sex'] == 'female') & (df['pclass'] == 1)
    mask_male_3 = (df['sex'] == 'male') & (df['pclass'] == 3)
    
    report_lines.append("### Survival Rates\n")
    report_lines.append("- **By Sex:**\n")
    for k, v in surv_sex.items():
        report_lines.append(f"  - {k}: {v:.2%}\n")
        
    report_lines.append("- **By Pclass:**\n")
    for k, v in surv_pclass.items():
        report_lines.append(f"  - Class {k}: {v:.2%}\n")
        
    report_lines.append("- **By Sex AND Pclass (Examples via Masking):**\n")
    report_lines.append(f"  - Female in 1st Class: {df[mask_female_1]['survived'].mean():.2%}\n")
    report_lines.append(f"  - Male in 3rd Class: {df[mask_male_3]['survived'].mean():.2%}\n")
    
    # Correlation Matrix
    corr_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
    corr_matrix = df[corr_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Correlation Matrix (6x6)")
    plt.tight_layout()
    corr_plot_path = os.path.join(PLOTS_DIR, 'correlation_matrix.png')
    plt.savefig(corr_plot_path)
    plt.close()
    
    # Top 2 feature pairs
    # Unstack and drop self-correlations and duplicates
    c = corr_matrix.abs().unstack()
    c = c[c < 1.0].drop_duplicates().sort_values(ascending=False)
    top_2 = c.head(2)
    
    report_lines.append("\n### Correlation Matrix Analysis\n")
    report_lines.append(f"**Strongest Pair 1:** {top_2.index[0][0]} & {top_2.index[0][1]} (|r| = {top_2.values[0]:.3f})\n")
    report_lines.append(f"**Strongest Pair 2:** {top_2.index[1][0]} & {top_2.index[1][1]} (|r| = {top_2.values[1]:.3f})\n")
    
    report_lines.append("\n**Interpretation:**\n")
    report_lines.append("- The strongest relationship is between **pclass** and **fare**. This makes intuitive sense: higher ticket classes (lower numerical pclass values) command significantly higher fares.\n")
    report_lines.append("- The second strongest relationship is between **sibsp** and **parch**. This indicates that passengers travelling with siblings/spouses were also highly likely to be travelling with parents/children, representing entire family units aboard.\n")
    
    # 4. Multivariate Data Story (4 Charts)
    report_lines.append("\n## Multivariate Data Story\n")
    
    # Chart 1: Survival by Sex and Pclass
    plt.figure(figsize=(8, 5))
    sns.barplot(x='pclass', y='survived', hue='sex', data=df)
    plt.title('Chart 1: Survival Rate by Passenger Class and Sex')
    plt.savefig(os.path.join(PLOTS_DIR, 'chart1_survival_sex_pclass.png'))
    plt.close()
    
    report_lines.append("### Chart 1: Survival Rate by Passenger Class and Sex\n")
    report_lines.append("This bar plot illustrates the compounded effect of gender and socio-economic status on survival. Females across all classes survived at much higher rates than males, heavily driven by the 'women and children first' protocol. However, 3rd class females still suffered significantly higher mortality than 1st and 2nd class females, exposing a harsh class divide.\n\n")

    # Chart 2: Age Distribution by Survival
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='age', hue='survived', fill=True, common_norm=False)
    plt.title('Chart 2: Age Distribution Density by Survival Status')
    plt.savefig(os.path.join(PLOTS_DIR, 'chart2_age_survival.png'))
    plt.close()
    
    report_lines.append("### Chart 2: Age Distribution Density by Survival Status\n")
    report_lines.append("This density plot compares the age distributions of those who survived versus those who perished. A noticeable spike exists for young children (age < 10) in the 'survived' category, highlighting prioritizing children during evacuation. Conversely, young adults (20-30) form the bulk of the casualties.\n\n")

    # Chart 3: Fare vs Age Scatterplot colored by Survival
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='fare', hue='survived', size='pclass', sizes=(150, 20), data=df, alpha=0.7)
    plt.title('Chart 3: Fare vs Age Scatterplot by Survival and Class')
    plt.ylim(0, 300) # clip extreme outliers for readability
    plt.savefig(os.path.join(PLOTS_DIR, 'chart3_fare_age_survival.png'))
    plt.close()
    
    report_lines.append("### Chart 3: Fare vs Age Scatterplot by Survival and Class\n")
    report_lines.append("This scatterplot reveals the interaction between age, fare paid, class size, and survival outcome. The highest fares are concentrated among middle-aged adults, who overwhelmingly survived (orange dots). The bottom-left cluster represents lower-fare, younger individuals who suffered widespread casualties.\n\n")

    # Chart 4: Pairplot of Key Variables
    pair_df = df[['survived', 'age', 'fare', 'pclass']].copy()
    pair_df['survived'] = pair_df['survived'].astype(str)
    sns.pairplot(pair_df, hue='survived', corner=True)
    plt.savefig(os.path.join(PLOTS_DIR, 'chart4_pairplot.png'))
    plt.close()
    
    report_lines.append("### Chart 4: Key Multivariate Pairplot\n")
    report_lines.append("This pair plot summarizes the pairwise distributions across age, fare, and passenger class, separated by survival. It confirms visually that high fares (and thus 1st class) act as a strong separator for survival. It also emphasizes the dense clustering of non-survivors at the lowest fare and highest pclass (3rd class) margins.\n\n")

    # 5. Standardization Sanity Check
    report_lines.append("## Standardization Sanity Check (EDA Only)\n")
    
    age_z = (df['age'] - df['age'].mean()) / df['age'].std()
    fare_z = (df['fare'] - df['fare'].mean()) / df['fare'].std()
    
    report_lines.append("- **Age Z-score check:** Mean = {:.10f}, Std = {:.5f}\n".format(age_z.mean(), age_z.std()))
    report_lines.append("- **Fare Z-score check:** Mean = {:.10f}, Std = {:.5f}\n".format(fare_z.mean(), fare_z.std()))
    
    # Save report
    with open(os.path.join(REPORTS_DIR, 'eda_report.md'), 'w') as f:
        f.writelines(report_lines)
        
    print("EDA completed successfully. Report saved to outputs/reports/eda_report.md")
    
    return df

if __name__ == "__main__":
    raw_df = load_data()
    cleaned_df = generate_eda_report(raw_df)
