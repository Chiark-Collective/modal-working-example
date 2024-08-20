import torch
from model import SimpleCNN

def generate_random_weights():
    model = SimpleCNN()
    torch.save(model.state_dict(), 'local_weights/initial_weights.pt')
    print("Random weights saved to local_weights/initial_weights.pt")

if __name__ == "__main__":
    generate_random_weights()
