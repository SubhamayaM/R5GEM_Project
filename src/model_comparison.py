import numpy as np
import matplotlib.pyplot as plt
from src.baseline_models import baseline_single, baseline_handover
from src.proposed_model import proposed_model

def run_comparison():
    NUM_STEPS = 200
    NUM_TOWERS = 5

    tower_positions = np.linspace(0, 1000, NUM_TOWERS)
    train_position = np.linspace(0, 1000, NUM_STEPS)

    # Run models
    b1 = baseline_single(train_position, tower_positions)
    b2 = baseline_handover(train_position, tower_positions)
    p = proposed_model(train_position, tower_positions)

    print("\nMODEL COMPARISON RESULTS:\n")

    print("Baseline 1 (Single Connection):")
    print(f"Throughput: {b1[0]:.2f}, Latency: {b1[1]:.4f}, Packet Loss: {b1[2]:.4f}\n")

    print("Baseline 2 (Handover):")
    print(f"Throughput: {b2[0]:.2f}, Latency: {b2[1]:.4f}, Packet Loss: {b2[2]:.4f}, Handovers: {b2[3]}\n")

    print("Proposed Model (AI + Multi-Connectivity):")
    print(f"Throughput: {p[0]:.2f}, Latency: {p[1]:.4f}, Packet Loss: {p[2]:.4f}\n")

    # Graphs
    models = ['Baseline1', 'Baseline2', 'Proposed']
    throughputs = [b1[0], b2[0], p[0]]
    latencies = [b1[1], b2[1], p[1]]

    plt.figure()
    plt.bar(models, throughputs)
    plt.title("Throughput Comparison")
    plt.savefig("results/throughput.png")
    plt.show()

    plt.figure()
    plt.bar(models, latencies)
    plt.title("Latency Comparison")
    plt.savefig("results/latency.png")
    plt.show()

    # Save summary
    with open("results/summary.txt", "w") as f:
        f.write("MODEL COMPARISON RESULTS\n\n")
        f.write(f"Baseline1: {b1}\n")
        f.write(f"Baseline2: {b2}\n")
        f.write(f"Proposed: {p}\n")