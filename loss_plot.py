import matplotlib.pyplot as plt


def get_loss(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()

    total_loss = list()
    retain_loss = list()
    forget_loss = list()

    for line in lines:
        if "[Train Summary]" in line:
            total_loss.append(
                float(line[line.index("Total Loss:") + 11 : line.index("Retain Loss:")])
            )
            retain_loss.append(
                float(
                    line[line.index("Retain Loss:") + 12 : line.index("Forget Loss:")]
                )
            )
            forget_loss.append(float(line[line.index("Forget Loss:") + 12 :]))
    return total_loss, retain_loss, forget_loss


if __name__ == "__main__":
    log_file = "./logs/experiment_1B_mix_dynamic_0.5/2024-12-03-1437.log"

    total_loss, retain_train_loss, forget_train_loss = get_loss(log_file)
    print(total_loss)
    print(retain_train_loss)
    print(forget_train_loss)

    plt.plot(total_loss, "-o", label="Total Train Loss")
    plt.plot(retain_train_loss, "-o", label="Retain Train Loss")
    plt.plot(forget_train_loss, "-o", label="Forget Train Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
