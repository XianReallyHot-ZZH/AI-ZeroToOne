import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, activation=None):
        self.learning_rate = np.float64(learning_rate)
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.activation = activation    # 激活函数 Can be 'relu', 'sigmoid', or None

    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        """Derivative of Sigmoid activation function"""
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _apply_activation(self, x):
        """Apply activation function to input"""
        if self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'sigmoid':
            return self._sigmoid(x)
        else:
            return x

    def _apply_activation_derivative(self, x):
        """Apply derivative of activation function to input"""
        if self.activation == 'relu':
            return self._relu_derivative(x)
        elif self.activation == 'sigmoid':
            return self._sigmoid_derivative(x)
        else:
            return 1        # Derivative is 1 when no activation

    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape

        # Initialize weights with float64
        self.weights = np.random.randn(n_features).astype(np.float64) * 0.01
        self.bias = np.float64(0)

        # Gradient descent
        prev_loss = float('inf')
        patience = 5
        min_change = 1e-5
        patience_counter = 0

        for epoch in range(self.n_iterations):
            # Linear model prediction
            linear_output = np.dot(X, self.weights) + self.bias
            y_predicted = self._apply_activation(linear_output)

            # Compute gradients with better numerical stability
            activation_derivative = self._apply_activation_derivative(linear_output)
            diff = y_predicted - y

            dw = np.float64(1 / n_samples) * np.dot(X.T, (diff * activation_derivative))
            db = np.float64(1 / n_samples) * np.sum(diff * activation_derivative)

            # Clip gradients to prevent explosion
            clip_threshold = 2.0
            dw = np.clip(dw, -clip_threshold, clip_threshold)
            db = np.clip(db, -clip_threshold, clip_threshold)

            # Update parameters with gradient checking
            if not (np.isnan(dw).any() or np.isnan(db)):
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            # Add early stopping
            current_loss = np.mean((y_predicted - y) ** 2)
            if abs(prev_loss - current_loss) < min_change:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
            prev_loss = current_loss

            # Print progress
            if epoch % 1000 == 0:
                print("Epoch {}: loss {}".format(epoch, current_loss))

    def predict(self, X):
        """Predict with optional activation"""
        inear_output = np.dot(X, self.weights) + self.bias
        return self._apply_activation(inear_output)


    def score(self, X, y):
        """Calculate R-squared score"""
        y_pred = self.predict(X)
        ss_total = np.sum((y -y.mean())**2)
        ss_residual = np.sum((y - y_pred)**2)
        return 1 - ss_residual / ss_total

