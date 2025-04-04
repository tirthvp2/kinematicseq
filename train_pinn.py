# train_pinn.py
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.g = nn.Parameter(torch.tensor(4.9))

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, inputs, predictions):
        x_i, y_i, v_x, v_y, t = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]
        x_f_pred, y_f_pred = predictions[:, 0], predictions[:, 1]

        x_f_phys = x_i + v_x * t
        y_f_phys = y_i + v_y * t - 0.5 * self.g * t * t

        phys_loss_x = torch.mean((x_f_pred - x_f_phys) ** 2)
        phys_loss_y = torch.mean((y_f_pred - y_f_phys) ** 2)
        return phys_loss_x + phys_loss_y

def train_pinn(data_path, epochs=15000, lr=0.0003, alpha=10.0):
    df = pd.read_csv(data_path)

    # Normalize inputs and targets with tighter scaling
    inputs = torch.tensor(df[['x_initial_position', 'y_initial_position',
                             'x_initial_velocity', 'y_initial_velocity', 'time']].values,
                          dtype=torch.float32)
    targets = torch.tensor(df[['x_final_position', 'y_final_position']].values,
                           dtype=torch.float32)

    input_mean = inputs.mean(dim=0)
    input_std = inputs.std(dim=0) + 1e-6
    inputs_norm = (inputs - input_mean) / input_std

    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0) + 1e-6
    targets_norm = (targets - target_mean) / target_std

    model = PINN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(inputs_norm)
        data_loss = torch.mean((predictions - targets_norm) ** 2)

        # Physics loss on unnormalized scale
        pred_unnorm = predictions * target_std + target_mean
        inputs_unnorm = inputs_norm * input_std + input_mean
        phys_loss = model.physics_loss(inputs_unnorm, pred_unnorm)

        loss = data_loss + alpha * phys_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Learned g: {model.g.item():.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'input_mean': input_mean,
        'input_std': input_std,
        'target_mean': target_mean,
        'target_std': target_std
    }, 'model/pinn_model.pth')
    print("Model and normalization params saved to 'pinn_model.pth'")
    return model, input_mean, input_std, target_mean, target_std

if __name__ == "__main__":
    train_pinn('data/projectile_data.csv')
