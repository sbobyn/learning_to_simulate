import argparse
import os
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="datasets/mpm88")
    parser.add_argument("--default_connectivity_radius", type=float, default=0.018)

    args = parser.parse_args()

    ds = np.load(os.path.join(args.datapath, f"positions_train.npz"))

    positions = []
    for f in ds:
        positions.append(ds[f])

    # compute bounds
    bounds = np.zeros((2, 2), dtype=np.float64)
    for pos in positions:
        # horizontal bounds
        bounds[0][0] = round(min(bounds[0][0], pos[:, 0].min()))
        bounds[0][1] = round(max(bounds[0][1], pos[:, 0].max()))
        # vertical bounds
        bounds[1][0] = round(min(bounds[1][0], pos[:, 1].min()))
        bounds[1][1] = round(max(bounds[1][1], pos[:, 1].max()))

    # compute velocities as finite difference
    velocities = []
    for i in range(len(positions)):
        velocities.append(np.diff(positions[i], axis=0))

    # compute mean and std of velocities
    x_velocities = [v[:, :, 0] for v in velocities]
    y_velocities = [v[:, :, 1] for v in velocities]

    vel_xmean = np.mean(np.concatenate(x_velocities, axis=1), dtype=np.float64)
    vel_ymean = np.mean(np.concatenate(y_velocities, axis=1), dtype=np.float64)

    vel_xstd = np.std(np.concatenate(x_velocities, axis=1), dtype=np.float64)
    vel_ystd = np.std(np.concatenate(y_velocities, axis=1), dtype=np.float64)

    # compute accelerations as finite difference
    accelerations = []
    for i in range(len(velocities)):
        accelerations.append(np.diff(velocities[i], axis=0))

    x_accelerations = [a[:, :, 0] for a in accelerations]
    y_accelerations = [a[:, :, 1] for a in accelerations]

    acc_xmean = np.mean(np.concatenate(x_accelerations, axis=1), dtype=np.float64)
    acc_ymean = np.mean(np.concatenate(y_accelerations, axis=1), dtype=np.float64)

    acc_xstd = np.std(np.concatenate(x_accelerations, axis=1), dtype=np.float64)
    acc_ystd = np.std(np.concatenate(y_accelerations, axis=1), dtype=np.float64)

    sequence_length = positions[0].shape[0]
    dim = positions[0].shape[2]

    metadata = {
        "bounds": bounds.tolist(),
        "vel_mean": [vel_xmean, vel_ymean],
        "vel_std": [vel_xstd, vel_ystd],
        "acc_mean": [acc_xmean, acc_ymean],
        "acc_std": [acc_xstd, acc_ystd],
        "dim": dim,
        "sequence_length": sequence_length,
        "default_connectivity_radius": args.default_connectivity_radius,
    }

    with open(os.path.join(args.datapath, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()
