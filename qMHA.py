# imports
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pennylane as qml

num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)

# quantum circuit
def circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

# calculates attention scores and applies them to the values
def attend(Q, K, V, batch_size, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output
   

# attention class
class QuantumMultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, num_qubits=num_qubits, n_qlayers=1, dev=dev):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        assert embed_dim == num_qubits, "Embedding dimension must match number of qubits"
        self.embed_dim = embed_dim

        self.q_layer = qml.QNode(circuit, dev, interface='torch')
        self.weight_shapes = {"weights":(n_qlayers, num_qubits)}

        self.K_linear = qml.qnn.TorchLayer(self.q_layer, weight_shapes=self.weight_shapes)
        self.Q_linear = qml.qnn.TorchLayer(self.q_layer, weight_shapes=self.weight_shapes)
        self.V_linear = qml.qnn.TorchLayer(self.q_layer, weight_shapes=self.weight_shapes)

        self.combine_heads = qml.qnn.TorchLayer(self.q_layer, weight_shapes=self.weight_shapes)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        assert embed_dim == self.embed_dim, "Input embedding dimension must match model's embedding dimension"

        K = [self.K_linear(x[:, i, :]) for i in range(seq_len)]
        Q = [self.Q_linear(x[:, i, :]) for i in range(seq_len)]
        V = [self.V_linear(x[:, i, :]) for i in range(seq_len)]

        K = torch.Tensor(pad_sequence(K))
        Q = torch.Tensor(pad_sequence(Q))
        V = torch.Tensor(pad_sequence(V))

        x = attend(Q, K, V, batch_size, mask)
        output = [self.combine_heads(x[:, t, :]) for t in range(seq_len)]
        output = torch.Tensor(pad_sequence(output))
        return output
