import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    np.random.seed(42)
    X, y = make_regression(n_samples=250, n_features=25, n_informative=6, noise=15.0, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        'OLS (Linear)': LinearRegression(),
        'Lasso (L1)': Lasso(alpha=0.6, random_state=42),
        'Ridge (L2)': Ridge(alpha=15.0, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.6, l1_ratio=0.5, random_state=42)
    }

    results = {}
    print(f"{'Model':<16} | {'Train MSE':<10} | {'Test MSE':<10} | {'Test R2':<8} | {'Non-Zero Coeffs'}")
    print("-" * 65)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_tr = model.predict(X_train)
        y_pred_te = model.predict(X_test)
        
        tr_mse = mean_squared_error(y_train, y_pred_tr)
        te_mse = mean_squared_error(y_test, y_pred_te)
        te_r2 = r2_score(y_test, y_pred_te)
        non_zero = np.sum(np.abs(model.coef_) > 1e-4)

        results[name] = {'te_mse': te_mse, 'te_r2': te_r2, 'coef': model.coef_}
        print(f"{name:<16} | {tr_mse:<10.2f} | {te_mse:<10.2f} | {te_r2:<8.3f} | {non_zero}/{len(model.coef_)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_indices = np.arange(X.shape[1])
    width = 0.2

    for i, (name, res) in enumerate(results.items()):
        axes[0].bar(x_indices + i * width, res['coef'], width=width, label=name)

    axes[0].set_title("Learned Feature Coefficients")
    axes[0].set_xlabel("Feature Index")
    axes[0].set_ylabel("Coefficient Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    model_names = list(results.keys())
    test_mses = [results[m]['te_mse'] for m in model_names]
    axes[1].bar(model_names, test_mses, color=['gray', 'orange', 'royalblue', 'forestgreen'])
    axes[1].set_title("Test MSE Comparison")
    axes[1].set_ylabel("Mean Squared Error")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
