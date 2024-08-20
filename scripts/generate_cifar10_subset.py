import torch
import torchvision
import torchvision.transforms as transforms

def generate_cifar10_subset(num_images=100):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    subset_indices = torch.randperm(len(trainset))[:num_images]
    subset_data = [trainset[i] for i in subset_indices]
    
    images = torch.stack([img for img, _ in subset_data])
    labels = torch.tensor([label for _, label in subset_data])
    
    torch.save((images, labels), 'local_data/cifar10_sample.pt')
    print(f"CIFAR-10 subset ({num_images} images) saved to local_data/cifar10_sample.pt")

if __name__ == "__main__":
    generate_cifar10_subset()
