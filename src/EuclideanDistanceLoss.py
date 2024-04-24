import torch


class EuclideanDistanceLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanDistanceLoss, self).__init__()

    def euclidean_distance(self, pred_coords, true_coords):
        """Custom Euclidean distance loss function."""
        return torch.sqrt(torch.sum((pred_coords - true_coords)**2))

    def forward(self, pred_coords, true_coords):
        """Compute the Euclidean distance loss."""
        return self.euclidean_distance(pred_coords, true_coords)
