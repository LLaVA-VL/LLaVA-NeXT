import torch
import random
class FIFOMemory:
    def __init__(self, max_size, tensor_shape, device='cuda'):
        self.max_size = max_size
        self.tensor_shape = tensor_shape
        self.device = device
        self.memory = torch.zeros((0, *tensor_shape), device = self.device)


    def add_tensor(self, tensor):
        if isinstance(tensor, list):
            tensor = torch.cat(tensor, dim=0)  # Convert list to tensor
        self.memory = torch.cat((self.memory, tensor), dim=0)
        if self.memory.shape[0] > self.max_size:
            self.memory = self.memory[-self.max_size:]

    def get_tensors(self):
        return self.memory


# fifo_memory = FIFOMemory(max_size=5, tensor_shape=(3, 384, 384), device='cpu')
#
# # Adding tensors to the FIFO memory
# for i in range(7):
#     tensor = torch.randn(1, 3, 384, 384)  # Example tensor with batch size 1
#     fifo_memory.add_tensor(tensor)
#     print(f"Added tensor {i + 1}, memory shape: {fifo_memory.get_tensors().shape}")
#
# # Retrieving tensors from the FIFO memory
# tensors = fifo_memory.get_tensors()
# print(f"Total tensors in memory: {tensors.shape}")



class KMeansMemory:
    def __init__(self, max_size, tensor_shape, max_iterations=100, device='cuda'):
        self.max_size = max_size
        self.tensor_shape = tensor_shape
        self.max_iterations = max_iterations
        self.device = device
        self.memory = torch.zeros((0, *tensor_shape), device=self.device)
        self.weights = torch.ones(0, device=self.device)

    def add_tensor(self, tensor):
        if isinstance(tensor, list):
            tensor = torch.cat(tensor, dim=0)  # Convert list to tensor
        self.memory = torch.cat((self.memory, tensor), dim=0)
        new_weights = torch.ones(tensor.shape[0], device=self.device)  # New weights, one for each tensor added
        self.weights = torch.cat((self.weights, new_weights), dim=0)
        if self.memory.shape[0] > self.max_size:
            self.memory, self.weights = self.weighted_kmeans(self.memory, self.max_size, self.max_iterations, self.weights)

    def weighted_kmeans(self, memory, max_size, max_iterations, weights):
        num_points = memory.shape[0]
        centroids = memory[random.sample(range(num_points), max_size)]
        # Flatten the memory and centroids tensors for distance computation
        memory_flat = memory.view(memory.shape[0], -1)  # Flatten each tensor (num_points, feature_dim)
        centroids_flat = centroids.view(centroids.shape[0], -1)  # Flatten centroids (max_size, feature_dim)
        weights_sum = torch.zeros(max_size, dtype=memory.dtype, device=memory.device)
        prev_assignment = None
        for _ in range(max_iterations):
            distances = torch.cdist(memory_flat, centroids_flat)  # (num_points, max_size)
            current_assignment = torch.argmin(distances, dim=1)

            if prev_assignment is not None and torch.equal(current_assignment, prev_assignment):
                break
            for j in range(max_size):
                assigned_points = memory[current_assignment == j]
                assigned_weights = weights[current_assignment == j]
                if assigned_points.shape[0] > 0:
                    centroids[j] = torch.sum(assigned_weights[:, None, None, None] * assigned_points, dim=0) / torch.sum(assigned_weights)
                    weights_sum[j] = torch.sum(assigned_weights)
                    print(weights_sum)
            prev_assignment = current_assignment.clone()

        new_weights = weights_sum  # Update the weights to the sum of weights assigned to each centroid after iterations

        return centroids[:max_size], new_weights

    def get_tensors(self):
        return self.memory


# kmeans_memory = KMeansMemory(max_size=5, tensor_shape=(3, 384, 384), device='cpu')
#
# # Adding tensors to the KMeans memory
# for i in range(7):
#     tensor = torch.randn(1, 3, 384, 384)  # Example tensor with batch size 1
#     kmeans_memory.add_tensor(tensor)
#     print(f"Added tensor {i + 1}, memory shape: {kmeans_memory.get_tensors().shape}")
#
# # Retrieving tensors from the KMeans memory
# tensors = kmeans_memory.get_tensors()
# print(f"Total tensors in memory: {tensors.shape}")
