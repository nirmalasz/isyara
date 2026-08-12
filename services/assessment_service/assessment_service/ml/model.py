try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None


if nn:
    class SignClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.25):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, num_classes)

        def forward(self, sequence):
            output, _ = self.lstm(sequence)
            last_step = output[:, -1, :]
            return self.fc(self.dropout(last_step))
else:
    class SignClassifier:  # pragma: no cover - placeholder when torch is unavailable
        pass
