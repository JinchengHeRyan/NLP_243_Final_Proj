import matplotlib.pyplot as plt


def get_loss(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()

    retain_train_loss = list()
    forget_train_loss = list()

    for line in lines:
        if "[Train Summary]" in line:
            retain_train_loss.append(
                float(
                    line[line.index("Retain Loss:") + 12 : line.index("Forget Loss:")]
                )
            )
            forget_train_loss.append(float(line[line.index("Forget Loss:") + 12 :]))
    return retain_train_loss, forget_train_loss


if __name__ == "__main__":
    log_file = "./logs/experiment_1B/2024-12-02-2316.log"

    retain_train_loss, forget_train_loss = get_loss(log_file)
    print(retain_train_loss)
    print(forget_train_loss)

    plt.plot(retain_train_loss, "-o", label="Retain Train Loss")
    plt.plot(forget_train_loss, "-o", label="Forget Train Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
