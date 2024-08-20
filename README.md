# CIFAR-10 Classification with PyTorch and Modal

This project demonstrates how to set up and run a deep learning training pipeline for CIFAR-10 image classification using PyTorch, MLflow, and TensorBoard, all orchestrated through Modal for cloud-based execution.

## Prerequisites

- Python 3.8+
- Poetry
- Docker (for local testing and development)
- Modal CLI installed and configured

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/cifar10-modal-project.git
   cd cifar10-modal-project
   ```

2. Install the project dependencies:
   ```
   make install
   ```

3. Generate test data:
   ```
   make generate-data
   ```

4. Build and push the Docker image (this is done automatically via GitHub Actions on push to main):
   ```
   docker build -t ghcr.io/yourusername/cifar10-modal-project:latest .
   docker push ghcr.io/yourusername/cifar10-modal-project:latest
   ```

## Usage

Run the main script using Modal:

```
modal run main.py [OPTIONS]
```

Options:
- `--learning-rate FLOAT`: Learning rate for the optimizer (default: 0.001)
- `--batch-size INTEGER`: Batch size for training (default: 32)
- `--num-epochs INTEGER`: Number of epochs to train (default: 5)
- `--run-tests-first`: Run tests before training
- `--upload-sample-data`: Upload sample data before training
- `--upload-initial-weights`: Upload initial weights before training

Example:
```
modal run main.py --learning-rate 0.01 --batch-size 64 --num-epochs 10 --run-tests-first --upload-sample-data --upload-initial-weights
```

## Project Structure

- `main.py`: Main script for running the training pipeline
- `model.py`: Definition of the SimpleCNN model
- `dataset.py`: Data loading utilities for CIFAR-10
- `Dockerfile`: Defines the container environment
- `Makefile`: Contains commands for common tasks
- `pyproject.toml`: Poetry configuration and dependencies
- `tests/`: Directory containing test files
- `scripts/`: Utility scripts for generating test data
- `local_data/`: Directory for storing local data files
- `local_weights/`: Directory for storing local weight files
- `.github/workflows/`: GitHub Actions workflow for building and pushing the Docker image

## Monitoring and Visualization

- MLflow: Access the MLflow UI using the URL provided in the console output
- TensorBoard: Access the TensorBoard UI using the URL provided in the console output

## Contributing

Feel free to submit issues or pull requests if you find any problems or have suggestions for improvements.

## License

This project is licensed under the MIT License.
