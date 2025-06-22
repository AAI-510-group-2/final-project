from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import numpy as np

def print_title(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

def print_sub_title(title):
    print("*" * 25 + title + "*" * 25 + "\n")


def print_model_performance(X_train, y_train, rf_train_pred, y_test, rf_test_pred, title, model):
    # Evaluate the model
    rf_train_r2 = r2_score(y_train, rf_train_pred)
    rf_test_r2 = r2_score(y_test, rf_test_pred)
    rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
    rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
    rf_test_mae = mean_absolute_error(y_test, rf_test_pred)

    print(title)
    print(f"Training R²: {rf_train_r2:.4f}")
    print(f"Testing R²: {rf_test_r2:.4f}")
    print(f"Training RMSE: ${rf_train_rmse:,.0f}")
    print(f"Testing RMSE: ${rf_test_rmse:,.0f}")
    print(f"Training MAE: ${rf_train_mae:,.0f}")
    print(f"Testing MAE: ${rf_test_mae:,.0f}")

    # Cross-validation
    rf_cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² (mean ± std): {rf_cv_scores.mean():.4f} ± {rf_cv_scores.std():.4f}")