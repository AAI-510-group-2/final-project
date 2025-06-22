

def handle_outliers(X, y):
    print("Handling extreme outliers...")

    # Define outlier boundaries (you can adjust these based on domain knowledge)
    lower_percentile = 0.01  # Remove bottom 1%
    upper_percentile = 0.99  # Remove top 1%

    lower_bound = y.quantile(lower_percentile)
    upper_bound = y.quantile(upper_percentile)

    print(f"Original salary range: ${y.min():,.0f} - ${y.max():,.0f}")
    print(
        f"Outlier bounds ({lower_percentile * 100}% - {upper_percentile * 100}%): ${lower_bound:,.0f} - ${upper_bound:,.0f}")

    # Filter outliers
    outlier_mask = (y >= lower_bound) & (y <= upper_bound)
    X_filtered = X[outlier_mask]
    y_filtered = y[outlier_mask]

    print(f"Samples removed: {len(y) - len(y_filtered)} ({(len(y) - len(y_filtered)) / len(y) * 100:.1f}%)")
    print(f"Final dataset shape: {X_filtered.shape}")
    print(f"Final salary range: ${y_filtered.min():,.0f} - ${y_filtered.max():,.0f}")

    # Update the variables for modeling
    X_res = X_filtered
    y_res = y_filtered

    return X_res, y_res