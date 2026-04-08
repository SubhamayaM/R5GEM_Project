import numpy as np

def proposed_model(train_position, tower_positions):
    throughput = []
    latency = []
    packet_loss = []

    for i, pos in enumerate(train_position):
        signals = [1 / (abs(pos - t) + 1) for t in tower_positions]

        # Multi-connectivity (top 2 towers)
        top2 = sorted(signals, reverse=True)[:2]
        combined_signal = sum(top2) / 2

        # Simple AI smoothing (prediction)
        if i > 0:
            combined_signal = (combined_signal + throughput[-1] / 100) / 2

        throughput.append(combined_signal * 150)
        latency.append(1 / combined_signal)
        packet_loss.append(1 - combined_signal)

    return np.mean(throughput), np.mean(latency), np.mean(packet_loss)