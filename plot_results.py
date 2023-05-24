import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    file_name = "./results/alexnet_cifar100_quantization.csv"
    data = pd.read_csv(file_name)
    plt.figure()
    plt.plot(data["NumBits"], data["Accuracy"])
    plt.xlabel("Number of Bits")
    plt.ylabel("Accuracy")
    plt.title("AlexNet50 on CIFAR100")
    plt.show()