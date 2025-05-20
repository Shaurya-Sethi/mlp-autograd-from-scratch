import numpy as np
from sklearn.model_selection import train_test_split
from tensor import Tensor
from mlp import MLP
from losses import cross_entropy_loss
from optimizer import AdamW, StepLR
from utils import load_and_preprocess_data, save_model

if __name__ == '__main__':
    print("Loading Dataset...")

    # NOTE:
    # This example expects a tabular CSV file with specific columns as per the fault diagnosis dataset.
    # If you use your own dataset, you MUST adjust the data loading, preprocessing, and
    # model input dimensions in `utils.py` to match your feature and label columns.
    #
    # See README for the expected format and instructions on adapting to custom datasets.

    csv_path = "data/fault-data.csv"
    X_np, y_np, label_encoder = load_and_preprocess_data(csv_path)  # Replace with your CSV file path if using a different dataset
    print("Dataset successfully validated and preprocessed.")

    X_train_np, X_test_np, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np
    )
    X_train = Tensor(X_train_np)
    X_test = Tensor(X_test_np)
    input_dim = X_train_np.shape[1]
    output_dim = len(np.unique(y_np))
    model = MLP(input_dim, output_dim)

    epochs = 500
    best_accuracy = 0.0
    patience = 10
    patience_counter = 0

    optimizer = AdamW(model.params, lr=0.005, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.7)

    print("Training MLP Model...")
    for epoch in range(1, epochs + 1):
        model.training = True
        logits = model(X_train)
        loss = cross_entropy_loss(logits, y_train)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if epoch % 10 == 0:
            model.training = False
            test_logits = model(X_test)
            preds = np.argmax(test_logits.data, axis=1)
            accuracy = np.mean(preds == y_test)

            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.data:.4f}, Test Accuracy: {accuracy:.4f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
                save_model(model, filename="mlp_best_model.pkl")
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly Stopping Triggered. Stopping Training.")
                break

    print(f"\nFinal Fine-Tuned MLP Accuracy: {best_accuracy:.4f}")
    save_model(model)
