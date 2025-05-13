import numpy as np

def create_sequences(features, sequence_length=10, skip_frames=1, anomaly_indices=None):
    sequences = []
    labels = []
    num_frames = len(features)
    
    print(f"Total frames: {num_frames}")
    
    for i in range(0, num_frames - sequence_length + 1, skip_frames):
        print(f"Creating sequence starting at index {i} (Sequence length: {sequence_length})")
        sequence = features[i:i + sequence_length]
        sequences.append(sequence)
        
        # Check if the sequence corresponds to an anomaly
        label = 0 if i in anomaly_indices else 1  # Anomalous sequences get label 0, else 1
        labels.append(label)
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Sequences shape: {sequences.shape}")
    print(f"Labels shape: {labels.shape}")
    
    return sequences, labels

# Example anomaly indices (you should provide your own anomaly detection logic)
anomaly_indices = [5, 20, 50, 70, 100]  # Example indices where anomalies occur

# Load the extracted features
features = np.load("features_test_Test001.npy")

# Convert the features into sequences for RNN input
sequence_length = 30  # Adjust as needed
sequences, labels = create_sequences(features, sequence_length, anomaly_indices=anomaly_indices)

# Save the sequences and labels
np.save("sequences_test_Test001.npy", sequences)
np.save("labels_test_Test001.npy", labels)
