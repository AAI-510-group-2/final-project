from utils import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

def outlier_detection_and_analysis(df):
    print_sub_title("Outlier Detection and Analysis")
    # Analyze salary outliers
    print("Salary Analysis:")
    print(f"Salary range: ${df['salary_usd'].min():,.0f} - ${df['salary_usd'].max():,.0f}")
    print(f"Salary median: ${df['salary_usd'].median():,.0f}")
    print(f"Salary mean: ${df['salary_usd'].mean():,.0f}")

    salary_outliers, sal_lower, sal_upper = detect_outliers_iqr(df, 'salary_usd')
    print(f"\nSalary outliers: {len(salary_outliers)} ({len(salary_outliers) / len(df) * 100:.1f}%)")
    print(f"Outlier bounds: ${sal_lower:,.0f} - ${sal_upper:,.0f}")

    # Visualize salary distribution before cleaning
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(df['salary_usd'], bins=50, alpha=0.7)
    plt.title('Salary Distribution (All Data)')
    plt.xlabel('Salary USD')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.boxplot(df['salary_usd'])
    plt.title('Salary Boxplot (All Data)')
    plt.ylabel('Salary USD')

    plt.subplot(1, 3, 3)
    # Remove top 5% outliers for better visualization
    upper_percentile = df['salary_usd'].quantile(0.95)
    filtered_salaries = df[df['salary_usd'] <= upper_percentile]['salary_usd']
    plt.hist(filtered_salaries, bins=50, alpha=0.7)
    plt.title('Salary Distribution (Bottom 95%)')
    plt.xlabel('Salary USD')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def handle_missing_values(df):
    print_sub_title("Handle Missing Values")
    df_clean = df.copy()

    print("Before cleaning:")
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")

    # Remove rows with missing target variable (salary_usd)
    df_clean = df_clean.dropna(subset=['salary_usd'])

    # Handle missing values in other important columns
    # For categorical variables, we can either drop or fill with mode
    # For numerical variables, we can fill with median/mean

    # Check which columns still have missing values
    missing_after_salary = df_clean.isnull().sum()
    print(f"\nAfter removing rows with missing salary:")
    print(f"Dataset shape: {df_clean.shape}")
    print("Remaining missing values:")
    print(missing_after_salary[missing_after_salary > 0])

    # Handle specific missing values based on data type and importance
    # Fill missing experience with median
    if 'years_experience' in df_clean.columns and df_clean['years_experience'].isnull().sum() > 0:
        median_exp = df_clean['years_experience'].median()
        df_clean['years_experience'].fillna(median_exp, inplace=True)

    # Fill missing categorical variables with mode or 'Unknown'
    categorical_cols_to_fill = ['experience_level', 'education_required', 'company_size', 'employment_type']
    for col in categorical_cols_to_fill:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
            df_clean[col].fillna(mode_value, inplace=True)

    # For skills, fill with 'Not specified'
    if 'required_skills' in df_clean.columns:
        # df_clean['required_skills'].fillna('Not specified', inplace=True)
        df_clean.fillna({'required_skills': 'Not specified'}, inplace=True)

    print(f"\nAfter handling missing values:")
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")

    return df_clean

def data_preparation(df):
    print_title("Data Preparation")
    print_sub_title("Convert date columns to datetime")
    date_columns = ['posting_date', 'application_deadline']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    print_sub_title("Check unique values in categorical columns to identify any cleaning needs")
    categorical_cols = ['employment_type', 'experience_level', 'education_required',
                        'company_size', 'industry', 'salary_currency']

    print("Unique values in categorical columns:")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\n{col}: {df[col].nunique()} unique values")
            print(df[col].value_counts().head(10))

    outlier_detection_and_analysis(df)
    df_clean = handle_missing_values(df)

    return df_clean

def data_splitting(X, y, test_size=0.2):
    print_title("Data Splitting")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=pd.cut(y, bins=5)
    )

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Target variable statistics:")
    print(f"  Train mean: ${y_train.mean():,.0f}")
    print(f"  Test mean: ${y_test.mean():,.0f}")

    return X_train, X_test, y_train, y_test