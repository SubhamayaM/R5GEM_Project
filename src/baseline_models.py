import numpy as np

# Signal strength function
def get_signal_strength(train_pos, tower_pos):
    return 1 / (abs(train_pos - tower_pos) + 1)

# ------------------------------
# Baseline Model 1: Single Connection
# ------------------------------
def baseline_single(train_position, tower_positions):
    throughput = []
    latency = []
    packet_loss = []

    for pos in train_position:
        signals = [get_signal_strength(pos, t) for t in tower_positions]
        best_signal = max(signals)

        throughput.append(best_signal * 100)
        latency.append(1 / best_signal)
        packet_loss.append(1 - best_signal)

    return np.mean(throughput), np.mean(latency), np.mean(packet_loss)

# ------------------------------
# Baseline Model 2: Basic Handover
# ------------------------------
def baseline_handover(train_position, tower_positions):
    current_tower = 0
    handovers = 0
    throughput = []
    latency = []
    packet_loss = []

    for pos in train_position:
        signals = [get_signal_strength(pos, t) for t in tower_positions]
        best_tower = np.argmax(signals)

        if best_tower != current_tower:
            handovers += 1
            current_tower = best_tower

        signal = signals[current_tower]

        throughput.append(signal * 120)
        latency.append(1 / signal)
        packet_loss.append(1 - signal)

    return np.mean(throughput), np.mean(latency), np.mean(packet_loss), handovers