# main.py
from pysr import PySRRegressor  # Import PySR first
try:
    from juliacall import Main as jl  # Import juliacall before torch
    jl.seval("using SymbolicRegression")
    jl.seval("using Statistics")
    print("Julia initialized successfully via juliacall.")
except ImportError:
    print("Warning: juliacall not found. PySR will attempt to initialize Julia automatically.")
except Exception as e:
    print(f"Julia initialization error: {e}. PySR will attempt to proceed.")

import torch  # Import torch after juliacall
import pandas as pd
from train_pinn import PINN


def load_pinn_model(model_path, data_path):
    checkpoint = torch.load(model_path)
    model = PINN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_mean = checkpoint['input_mean']
    input_std = checkpoint['input_std']
    target_mean = checkpoint['target_mean']
    target_std = checkpoint['target_std']

    df = pd.read_csv(data_path)
    inputs = torch.tensor(df[['x_initial_position', 'y_initial_position',
                             'x_initial_velocity', 'y_initial_velocity', 'time']].values,
                          dtype=torch.float32)
    inputs_norm = (inputs - input_mean) / input_std
    with torch.no_grad():
        predictions_norm = model(inputs_norm)
        predictions = predictions_norm * target_std + target_mean
    return model, df, predictions


def symbolic_regression(df):
    X = df[['x_initial_position', 'y_initial_position', 'x_initial_velocity', 'y_initial_velocity', 'time']].values
    y_x = df['x_final_position'].values
    y_y = df['y_final_position'].values

    # PySR model for x_final_position with built-in loss
    model_x = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        model_selection="best",
        elementwise_loss="L2DistLoss()",  # Use built-in MSE loss
        progress=False
    )
    model_x.fit(X, y_x)
    # print("\nSymbolic Formula for x_final_position:")
    # print(model_x)

    # PySR model for y_final_position with built-in loss
    model_y = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        model_selection="best",
        elementwise_loss="L2DistLoss()",  # Use built-in MSE loss
        progress=False
    )
    model_y.fit(X, y_y)
    # print("\nSymbolic Formula for y_final_position:")
    # print(model_y)


if __name__ == "__main__":
    model, df, predictions = load_pinn_model('model/pinn_model.pth', 'data/projectile_data.csv')

    print("\nSample Predictions vs Actual:")
    for i in range(5):
        print(f"Data point {i}:")
        print(f"Predicted: x_f={predictions[i, 0]:.2f}, y_f={predictions[i, 1]:.2f}")
        print(f"Actual: x_f={df['x_final_position'].iloc[i]:.2f}, y_f={df['y_final_position'].iloc[i]:.2f}")

    print(f"\nLearned gravity (g): {model.g.item():.4f} m/s² (True value: 4.9 m/s²)")

    symbolic_regression(df)
