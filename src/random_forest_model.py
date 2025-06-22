from utils import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def random_forest_train(X_train, X_test, y_train, y_test):
    print_title("Random Forest Model")

    # Initialize and train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    rf_train_pred = rf_model.predict(X_train)
    rf_test_pred = rf_model.predict(X_test)

    print_model_performance(X_train, y_train, rf_train_pred, y_test, rf_test_pred, "Random Forest Performance:", rf_model)

def random_forest_optimization(X_train, X_test, y_train, y_test):
    print_title("Random Forest Model Optimization")
    # Define parameter grid for optimization
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize model
    rf = RandomForestRegressor(random_state=42)

    # Grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)

    # Fit model
    grid_search.fit(X_train, y_train)

    # Evaluate best model
    best_rf = grid_search.best_estimator_

    print("Best Parameters:", grid_search.best_params_)
    # Make predictions
    rf_train_pred = best_rf.predict(X_train)
    rf_test_pred = best_rf.predict(X_test)
    print_model_performance(X_train, y_train, rf_train_pred, y_test, rf_test_pred, "Random Forest Performance:", best_rf)