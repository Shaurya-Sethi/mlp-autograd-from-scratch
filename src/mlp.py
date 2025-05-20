from layers import Linear, BatchNorm1d, selu, dropout

class MLP:
    """
    Multi-Layer Perceptron Model.
    """
    def __init__(self, input_dim, output_dim):
        self.fc1 = Linear(input_dim, 128)
        self.bn1 = BatchNorm1d(128)
        self.fc2 = Linear(128, 256)
        self.bn2 = BatchNorm1d(256)
        self.fc3 = Linear(256, 128)
        self.bn3 = BatchNorm1d(128)
        self.fc4 = Linear(128, output_dim)
        self.dropout_p = 0.3
        self.training = True
        self.params = (
            self.fc1.params + self.bn1.params +
            self.fc2.params + self.bn2.params +
            self.fc3.params + self.bn3.params +
            self.fc4.params
        )

    def forward(self, x):
        x = self.fc1(x); x = self.bn1(x); x = selu(x); x = dropout(x, self.dropout_p, self.training)
        x = self.fc2(x); x = self.bn2(x); x = selu(x); x = dropout(x, self.dropout_p, self.training)
        x = self.fc3(x); x = self.bn3(x); x = selu(x)
        x = self.fc4(x)
        return x

    def __call__(self, x):
        return self.forward(x)