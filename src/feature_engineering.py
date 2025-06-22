from utils import *
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_skill_features(skills_text):
    """Extract common AI/ML skills from the skills text"""
    if pd.isna(skills_text) or skills_text == 'Not specified':
        return {
            'has_python': 0, 'has_sql': 0, 'has_machine_learning': 0,
            'has_deep_learning': 0, 'has_tensorflow': 0, 'has_pytorch': 0,
            'has_aws': 0, 'has_azure': 0, 'has_gcp': 0, 'skill_count': 0
        }

    skills_lower = str(skills_text).lower()

    # Define skill patterns
    skill_patterns = {
        'has_python': r'\bpython\b',
        'has_sql': r'\bsql\b',
        'has_machine_learning': r'\b(machine learning|ml)\b',
        'has_deep_learning': r'\b(deep learning|neural network)\b',
        'has_tensorflow': r'\btensorflow\b',
        'has_pytorch': r'\bpytorch\b',
        'has_aws': r'\b(aws|amazon web services)\b',
        'has_azure': r'\bazure\b',
        'has_gcp': r'\b(gcp|google cloud)\b'
    }

    features = {}
    for skill, pattern in skill_patterns.items():
        features[skill] = 1 if re.search(pattern, skills_lower) else 0

    # Count total number of skills (split by common delimiters)
    skill_list = re.split(r'[,;|]', skills_text)
    features['skill_count'] = len([s.strip() for s in skill_list if s.strip()])

    return features


def categorize_seniority(row):
    """Categorize job seniority based on title and experience"""
    title = str(row.get('job_title', '')).lower()
    years_exp = row.get('years_experience', 0)

    if any(word in title for word in ['senior', 'lead', 'principal', 'staff']):
        return 'Senior'
    elif any(word in title for word in ['junior', 'entry', 'associate']):
        return 'Junior'
    elif any(word in title for word in ['manager', 'director', 'head', 'chief']):
        return 'Management'
    elif years_exp >= 5:
        return 'Senior'
    elif years_exp <= 2:
        return 'Junior'
    else:
        return 'Mid-level'


def extract_location_features(location):
    """Extract country and region from location"""
    if pd.isna(location):
        return 'Unknown', 'Unknown'

    location_str = str(location).strip()

    # Common patterns for extracting country
    if ',' in location_str:
        parts = location_str.split(',')
        country = parts[-1].strip()
        city = parts[0].strip()
    else:
        country = location_str
        city = location_str

    # Categorize by major regions/countries
    tech_hubs = ['United States', 'USA', 'US', 'Canada', 'United Kingdom', 'UK',
                 'Germany', 'Netherlands', 'Singapore', 'Australia', 'Ireland']

    if any(hub.lower() in country.lower() for hub in tech_hubs):
        region = 'Major Tech Hub'
    else:
        region = 'Other'

    return country, region

def encode_categorical_features(df):
    df_ml = df.copy()

    # Define categorical columns to encode
    categorical_cols = ['employment_type', 'experience_level', 'education_required',
                        'company_size', 'industry', 'seniority_level', 'region']

    # Define numerical columns to keep
    numerical_cols = ['years_experience', 'remote_ratio', 'benefits_score',
                      'job_description_length', 'skill_count'] + \
                     [col for col in df_ml.columns if col.startswith('has_')]

    # Label encode categorical variables with low cardinality
    label_encoders = {}
    for col in categorical_cols:
        if col in df_ml.columns:
            le = LabelEncoder()
            df_ml[col + '_encoded'] = le.fit_transform(df_ml[col].astype(str))
            label_encoders[col] = le

    # Create dummy variables for categorical columns with moderate cardinality
    # (We'll use this approach for flexibility in model interpretation)
    categorical_cols_for_dummies = ['employment_type', 'experience_level', 'education_required',
                                    'company_size', 'seniority_level', 'region']

    df_ml_dummies = pd.get_dummies(df_ml, columns=categorical_cols_for_dummies, prefix=categorical_cols_for_dummies)

    print("Categorical encoding completed.")
    print(f"Dataset shape after encoding: {df_ml_dummies.shape}")
    print(f"\nNew encoded columns (sample):")
    encoded_cols = [col for col in df_ml_dummies.columns if any(cat in col for cat in categorical_cols_for_dummies)]
    print(encoded_cols[:10])

    return df_ml_dummies, encoded_cols, categorical_cols

def remove_non_predictive_features(df, categorical_cols):
    # Columns to exclude from modeling
    columns_to_exclude = ['job_id', 'company_name', 'job_title', 'posting_date',
                          'application_deadline', 'company_location', 'employee_residence',
                          'required_skills', 'salary_currency', 'country', 'salary_range']

    # Also exclude original categorical columns that we've encoded
    columns_to_exclude.extend(categorical_cols)

    # Select features for modeling
    feature_columns = [col for col in df.columns
                       if col not in columns_to_exclude and col != 'salary_usd']

    # Create final feature matrix and target variable
    X = df[feature_columns]
    y = df['salary_usd']

    print("Feature selection completed:")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Target variable (salary) range: ${y.min():,.0f} - ${y.max():,.0f}")

    print(f"\nFeature columns (first 15):")
    print(X.columns[:15].tolist())

    # Check for any remaining missing values
    print(f"\nMissing values in features: {X.isnull().sum().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")

    # Basic correlation analysis with target
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\nTop 10 features correlated with salary:")
    print(correlations.head(10))

    return X, y

def feature_engineering(df):
    print_title("Feature Engineering")
    df_features = df.copy()
    print_sub_title("Apply skill extraction")
    skill_df = None
    if 'required_skills' in df_features.columns:
        skill_features = df_features['required_skills'].apply(extract_skill_features)
        skill_df = pd.DataFrame(skill_features.tolist())
        df_features = pd.concat([df_features, skill_df], axis=1)

    print("Skill features extracted:")
    print(skill_df.head())

    print_sub_title("Create additional derived features")
    df_features['seniority_level'] = df_features.apply(categorize_seniority, axis=1)

    print_sub_title("Apply location feature extraction")
    if 'company_location' in df_features.columns:
        location_features = df_features['company_location'].apply(lambda x: extract_location_features(x))
        df_features['country'] = [loc[0] for loc in location_features]
        df_features['region'] = [loc[1] for loc in location_features]

    print_sub_title("Create salary bins for analysis")
    df_features['salary_range'] = pd.cut(df_features['salary_usd'],
                                         bins=[0, 50000, 75000, 100000, 150000, np.inf],
                                         labels=['<50K', '50K-75K', '75K-100K', '100K-150K', '>150K'])

    print("New features created:")
    print(f"Seniority levels: {df_features['seniority_level'].value_counts()}")
    print(f"\nRegions: {df_features['region'].value_counts()}")
    print(f"\nSalary ranges: {df_features['salary_range'].value_counts()}")

    print_sub_title("Encode Categorical Variables")
    df_ml_dummies, encoded_cols, categorical_cols = encode_categorical_features(df_features)
    X, y = remove_non_predictive_features(df_ml_dummies, categorical_cols)

    return X, y, df_ml_dummies
