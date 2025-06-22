from utils import *
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def xgboost_model_train(X_train, X_test, y_train, y_test):
    # XGBoost Regressor
    print_title("XGBoost Model")

    # Initialize and train XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Make predictions
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)

    print_model_performance(X_train, y_train, xgb_train_pred, y_test, xgb_test_pred, "XGBoost Performance:", xgb_model)

def xgboost_model_optimization(X_train, X_test, y_train, y_test):
    print_title("XGBoost Model Optimization")
    # Define parameter grid for optimization
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [2, 4, 7, 9],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }

    # Initialize model
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, colsample_bytree=0.8)

    # Train the model
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring='r2', cv=5, n_jobs=-1, verbose=1)

    # Fit model
    grid_search.fit(X_train, y_train)

    # Evaluate best model
    best_xgb = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    # Make predictions
    xgb_train_pred = best_xgb.predict(X_train)
    xgb_test_pred = best_xgb.predict(X_test)

    print_model_performance(X_train, y_train, xgb_train_pred, y_test, xgb_test_pred, "XGBoost Performance:", best_xgb)