from utils import *

def final_validation_and_summary(original_df, cleaned_df, final_df, X, y):
    print_title("Data cleaning and preparation summary")

    categorical_cols_for_dummies = ['employment_type', 'experience_level', 'education_required',
                                    'company_size', 'seniority_level', 'region']
    print(f"Original dataset shape: {original_df.shape}")
    print(f"Cleaned dataset shape: {cleaned_df.shape}")
    print(f"Final ML dataset shape: {final_df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")

    print(f"\nData Quality Checks:")
    print(f"✓ Missing values handled: {cleaned_df.isnull().sum().sum()} remaining")
    print(f"✓ Duplicates removed: {original_df.duplicated().sum()} found in original")
    print(f"✓ Feature engineering completed: {len([c for c in X.columns if 'has_' in c])} skill features")
    print(
        f"✓ Categorical encoding completed: {len([c for c in X.columns if '_' in c and any(cat in c for cat in categorical_cols_for_dummies)])} dummy variables")

    print(f"\nDataset Statistics:")
    print(f"Salary range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(f"Average salary: ${y.mean():,.0f}")
    print(f"Median salary: ${y.median():,.0f}")

    print(f"\nKey Features Available:")
    print("- Job characteristics: title, experience, education, employment type")
    print("- Company information: size, industry, location, benefits")
    print("- Skills: Python, SQL, ML, DL, cloud platforms, etc.")
    print("- Location: country, region classification")
    print("- Seniority: derived from title and experience")

    print(f"\nReady for:")
    print("✓ Exploratory Data Analysis")
    print("✓ Predictive Modeling (Random Forest, XGBoost)")
    print("✓ Feature Importance Analysis")
    print("✓ Salary Prediction")

    # Optional: Save cleaned datasets
    save_data = False  # Set to True if you want to save the cleaned data

    if save_data:
        cleaned_df.to_csv('ai_job_dataset_cleaned.csv', index=False)
        final_df.to_csv('ai_job_dataset_ml_ready.csv', index=False)

        # Save feature matrix and target separately
        X.to_csv('features.csv', index=False)
        y.to_csv('target.csv', index=False)

        print(f"\n✓ Cleaned datasets saved to CSV files")

    print("\n" + "=" * 60)
    print("Data cleaning and preparation completed successfully!")
    print("You can now proceed with exploratory data analysis and modeling.")
    print("=" * 60)