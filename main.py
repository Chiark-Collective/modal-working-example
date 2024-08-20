"""
Main script for CIFAR-10 image classification using PyTorch and Modal.

This script sets up and runs a deep learning training pipeline on Modal,
including data and weight management, MLflow logging, and TensorBoard visualization.
It uses a simple CNN model to classify CIFAR-10 images.
"""

import os
import logging
from typing import Optional
import typer
from modal import Image, Mount, Stub, Volume, web_endpoint
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter
from model import SimpleCNN
from dataset import get_cifar10_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up Modal stub and volumes
stub = Stub("cifar10-training")
data_volume = Volume.persisted("cifar10-data-volume")
weights_volume = Volume.persisted("cifar10-weights-volume")
mlflow_volume = Volume.persisted("cifar10-mlflow-volume")

# Define the Docker image
image = Image.from_dockerfile("Dockerfile")

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

@stub.function(gpu="A100", image=image, volumes={"/data": data_volume, "/weights": weights_volume, "/mlflow": mlflow_volume})
def train(hyperparameters: dict):
    """
    Train the CIFAR-10 model using the provided hyperparameters.

    Args:
        hyperparameters (dict): A dictionary containing the hyperparameters for training.

    This function sets up the training environment, initializes the model,
    and runs the training loop while logging metrics to MLflow and TensorBoard.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        for key, value in hyperparameters.items():
            mlflow.log_param(key, value)
        
        writer = SummaryWriter(log_dir='/mlflow/tensorboard_logs')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        model = SimpleCNN().to(device)
        
        if os.path.exists("/weights/initial_weights.pt"):
            logger.info("Loading initial weights")
            model.load_state_dict(torch.load("/weights/initial_weights.pt"))
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=hyperparameters['learning_rate'], momentum=0.9)
        
        trainloader, testloader = get_cifar10_data(batch_size=hyperparameters['batch_size'])
        
        logger.info("Starting training")
        for epoch in range(hyperparameters['num_epochs']):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i % 200 == 199:
                    avg_loss = running_loss / 200
                    logger.info(f'Epoch {epoch + 1}, Batch {i + 1}: Loss {avg_loss:.3f}')
                    mlflow.log_metric("loss", avg_loss, step=epoch * len(trainloader) + i)
                    writer.add_scalar('training loss', avg_loss, epoch * len(trainloader) + i)
                    running_loss = 0.0
        
        logger.info("Finished training")
        
        torch.save(model.state_dict(), '/weights/final_model.pt')
        mlflow.pytorch.log_model(model, "model")
        
        writer.close()

@stub.function(image=image, volumes={"/mlflow": mlflow_volume}, keep_warm=1)
@web_endpoint(method="GET")
def mlflow_server():
    """Start the MLflow tracking server."""
    from mlflow.server import app
    return app

@stub.function(image=image, volumes={"/mlflow": mlflow_volume}, keep_warm=1)
@web_endpoint(method="GET")
def tensorboard_server():
    """Start the TensorBoard server."""
    import tensorboard
    from tensorboard import program
    tb = program.TensorBoard()
    tb.configure(logdir='/mlflow/tensorboard_logs')
    return tb.launch()

@stub.function(image=image)
def run_tests():
    """
    Run unit tests and CUDA availability check on the remote environment.

    This function executes 'make test' on the remote machine and checks for CUDA availability.
    """
    import subprocess
    
    # Run make test
    result = subprocess.run(['make', 'test'], capture_output=True, text=True)
    print("Make test output:")
    print(result.stdout)
    print(result.stderr)
    
    # Check CUDA availability
    import torch
    print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}")

    return result.returncode == 0 and torch.cuda.is_available()

@stub.function(volume=data_volume)
def upload_data(local_path: str, remote_path: str) -> bool:
    """
    Upload data from local path to the data volume.

    Args:
        local_path (str): Path to the local data file.
        remote_path (str): Path where the file will be stored in the data volume.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    try:
        with open(local_path, 'rb') as local_file:
            with open(remote_path, 'wb') as remote_file:
                remote_file.write(local_file.read())
        logger.info(f"Successfully uploaded {local_path} to {remote_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to {remote_path}: {str(e)}")
        return False

@stub.function(volume=weights_volume)
def upload_weights(local_path: str, remote_path: str) -> bool:
    """
    Upload weights from local path to the weights volume.

    Args:
        local_path (str): Path to the local weights file.
        remote_path (str): Path where the file will be stored in the weights volume.

    Returns:
        bool: True if upload was successful, False otherwise.
    """
    try:
        with open(local_path, 'rb') as local_file:
            with open(remote_path, 'wb') as remote_file:
                remote_file.write(local_file.read())
        logger.info(f"Successfully uploaded {local_path} to {remote_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_path} to {remote_path}: {str(e)}")
        return False

def main(
    learning_rate: float = typer.Option(0.001, help="Learning rate for the optimizer"),
    batch_size: int = typer.Option(32, help="Batch size for training"),
    num_epochs: int = typer.Option(5, help="Number of epochs to train"),
    run_tests_first: bool = typer.Option(False, help="Run tests before training"),
    upload_sample_data: bool = typer.Option(False, help="Upload sample data before training"),
    upload_initial_weights: bool = typer.Option(False, help="Upload initial weights before training"),
):
    """
    Main entry point for the CIFAR-10 training pipeline.

    This function orchestrates the entire training process, including optional testing,
    data and weight uploads, and the actual training run.
    """
    if run_tests_first:
        logger.info("Running tests...")
        tests_passed = run_tests.remote()
        if not tests_passed:
            logger.error("Tests failed. Aborting training.")
            return

    if upload_sample_data:
        logger.info("Uploading sample data...")
        upload_success = upload_data.remote("local_data/cifar10_sample.pt", "/data/cifar10_sample.pt")
        if not upload_success:
            logger.error("Failed to upload sample data. Aborting training.")
            return

    if upload_initial_weights:
        logger.info("Uploading initial weights...")
        upload_success = upload_weights.remote("local_weights/initial_weights.pt", "/weights/initial_weights.pt")
        if not upload_success:
            logger.error("Failed to upload initial weights. Aborting training.")
            return

    hyperparameters = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }
    
    logger.info(f"Starting training with hyperparameters: {hyperparameters}")
    train.remote(hyperparameters)
    
    logger.info(f"MLflow server is running at: {mlflow_server.web_url}")
    logger.info(f"TensorBoard server is running at: {tensorboard_server.web_url}")
    input("Press Enter to shut down the servers...")

if __name__ == "__main__":
    typer.run(main)
