import matplotlib.pyplot as plt
import re


def get_loss(log_file):
    """
    Parse the log file and extract losses for plotting.

    Args:
        log_file (str): Path to the log file.

    Returns:
        dict: A dictionary containing losses for 'ran', 'retain', 'forget', and 'pgd'.
    """
    losses = {"ran": [], "retain": [], "forget": [], "pgd": []}

    with open(log_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "[Train Summary]" in line:
            # Extract the losses using regex
            match = re.search(
                r"Ran Loss:([\d.]+)\s+Retain Loss:([\d.]+)\s+Forget Loss:([\d.]+)\s+PGD Loss:([\d.]+)",
                line,
            )
            if match:
                losses["ran"].append(float(match.group(1)))
                losses["retain"].append(float(match.group(2)))
                losses["forget"].append(float(match.group(3)))
                losses["pgd"].append(float(match.group(4)))

    return losses


if __name__ == "__main__":
    log_file = "./logs/gd_test/2024-12-03-1323.log"

    # Parse the log file
    losses = get_loss(log_file)

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(losses["ran"], "-o", label="Ran Loss")
    plt.plot(losses["retain"], "-o", label="Retain Loss")
    plt.plot(losses["forget"], "-o", label="Forget Loss")
    plt.plot(losses["pgd"], "-o", label="PGD Loss")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("./plots/loss_plot-12031323.png")
