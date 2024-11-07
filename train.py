import argparse
import os

import torch
import torch_geometric as pyg
from tqdm import tqdm

import dataset
import model
import rollout
import visualize


def oneStepMSE(simulator, dataloader, metadata, noise):
    """Returns two values, loss and MSE"""
    total_loss = 0.0
    total_mse = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        scale = torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise**2)
        for data in dataloader:
            pred = simulator(data)
            mse = ((pred - data.y) * scale) ** 2
            mse = mse.sum(dim=-1).mean()
            loss = ((pred - data.y) ** 2).mean()
            total_mse += mse.item()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count, total_mse / batch_count


def rolloutMSE(simulator, dataset, noise, metadata):
    total_loss = 0.0
    batch_count = 0
    simulator.eval()
    with torch.no_grad():
        for rollout_data in dataset:
            rollout_out = rollout.rollout(simulator, rollout_data, metadata, noise)
            rollout_out = rollout_out.permute(1, 0, 2)
            loss = (rollout_out - rollout_data["position"]) ** 2
            loss = loss.sum(dim=-1).mean()
            total_loss += loss.item()
            batch_count += 1
    return total_loss / batch_count


def train(
    params, simulator, train_loader, valid_loader, valid_rollout_dataset, metadata
):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=params["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / 5e6)
    )

    # recording loss curve
    train_loss_list = []
    eval_loss_list = []
    onestep_mse_list = []
    rollout_mse_list = []
    total_step = 0

    for i in range(params["epoch"]):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {i}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data = data.cuda()
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "avg_loss": total_loss / batch_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            total_step += 1
            train_loss_list.append((total_step, loss.item()))

            # evaluation
            if total_step % params["eval_interval"] == 0:
                simulator.eval()
                eval_loss, onestep_mse = oneStepMSE(
                    simulator, valid_loader, metadata, params["noise"]
                )
                eval_loss_list.append((total_step, eval_loss))
                onestep_mse_list.append((total_step, onestep_mse))
                tqdm.write(f"\nEval: Loss: {eval_loss}, One Step MSE: {onestep_mse}")
                simulator.train()

            # do rollout on valid set
            if total_step % params["rollout_interval"] == 0:
                simulator.eval()
                rollout_mse = rolloutMSE(
                    simulator, valid_rollout_dataset, params["noise"], metadata
                )
                rollout_mse_list.append((total_step, rollout_mse))
                tqdm.write(f"\nEval: Rollout MSE: {rollout_mse}")
                simulator.train()

            if total_step % params["vis_interval"] == 0:
                simulator.eval()
                rollout_data = valid_rollout_dataset[0]
                rollout_out = rollout.rollout(
                    simulator, rollout_data, metadata, params["noise"]
                )
                rollout_out = rollout_out.permute(1, 0, 2)
                anim = visualize.visualize_pair(
                    rollout_data["particle_type"],
                    rollout_out,
                    rollout_data["position"],
                    metadata,
                )
                video_path = os.path.join(
                    params["output_path"], f"rollout_{total_step}.mp4"
                )
                anim.save(video_path)
                tqdm.write(f"\nSaved video to {video_path}")
                simulator.train()

            # save model
            if total_step % params["save_interval"] == 0:
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    os.path.join(
                        os.path.join(params["output_path"], "models"),
                        f"checkpoint_{total_step}.pt",
                    ),
                )
    return train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--noise", type=float, default=3e-4)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--vis_interval", type=int, default=100000)
    parser.add_argument("--save_interval", type=int, default=100000)
    parser.add_argument("--rollout_interval", type=int, default=100000)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "models"), exist_ok=True)

    train_dataset = dataset.OneStepDataset(
        args.data_path, "train", noise_std=args.noise
    )
    valid_dataset = dataset.OneStepDataset(
        args.data_path, "valid", noise_std=args.noise
    )
    train_loader = pyg.loader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    valid_loader = pyg.loader.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    rollout_dataset = dataset.RolloutDataset(args.data_path, "valid")

    simulator = model.LearnedSimulator(hidden_size=32, n_mp_layers=2)
    if torch.cuda.is_available():
        simulator = simulator.cuda()

    params = vars(args)
    train_loss_list, eval_loss_list, onestep_mse_list, rollout_mse_list = train(
        params,
        simulator,
        train_loader,
        valid_loader,
        rollout_dataset,
        train_dataset.metadata,
    )


if __name__ == "__main__":
    main()
