# Import necessary modules from Flower framework
import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents


def server_fn(context: Context):
    # Define the strategy for federated averaging (FedAvg)
    strategy = fl.server.strategy.FedAvg(
        # Number of federated rounds
        fraction_fit=1.0,  # Use all clients in each round
        fraction_evaluate=0.0, # No evaluation during training (can be customized)
        min_fit_clients=5,  # Minimum number of clients to participate
        min_evaluate_clients=2, # Minimum clients for evaluation
        min_available_clients=5,  # Minimum clients to start training
        evaluate_fn=None,  # Optional evaluation function (left as None for simplicity)
        # Custom aggregation function can be added here for differential privacy
    )


    return ServerAppComponents(strategy=strategy)
# Start the Flower server with the specified strategy
app = ServerApp(server_fn=server_fn)



