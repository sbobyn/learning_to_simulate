import torch

import dataset
import model

BOUNDARY_PARTICLE = 3


def rollout(model, data, metadata, noise_std):
    device = next(model.parameters()).device
    model.eval()
    window_size = model.window_size + 1
    total_time = data["position"].size(0)
    traj = data["position"][:window_size]
    traj = traj.permute(1, 0, 2)
    particle_type = data["particle_type"]

    for time in range(total_time - window_size):
        with torch.no_grad():
            graph = dataset.preprocess(
                particle_type, traj[:, -window_size:], None, metadata, 0.0
            )
            graph = graph.to(device)
            acceleration = model(graph).cpu()
            acceleration = acceleration * torch.sqrt(
                torch.tensor(metadata["acc_std"]) ** 2 + noise_std**2
            ) + torch.tensor(metadata["acc_mean"])

            # Zero out the acceleration for boundary particles
            mask = data["particle_type"] != BOUNDARY_PARTICLE
            acceleration = acceleration * mask.unsqueeze(-1).float()

            recent_position = traj[:, -1]
            recent_velocity = recent_position - traj[:, -2]
            new_velocity = recent_velocity + acceleration
            new_position = recent_position + new_velocity
            traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

    return traj
