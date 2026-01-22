import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
from math import ceil
from typing import Optional, Tuple, Dict


#########################################################
## LOADER UTILS
#########################################################


class StratifiedBatchSampler:
    """
    Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        elif isinstance(y, list):
            y = np.array(y)
        assert len(y.shape) == 1, 'label array must be 1D'

        min_class_size = np.bincount(y).min()
        n_batches = int(len(y) / batch_size)
        n_splits = min(n_batches, min_class_size)  # <= fix
        n_splits = max(n_splits, 2)  # at least 2 splits
        
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.n_batches = n_splits  # adjusted

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0, int(1e8), size=()).item()
        for _, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return self.n_batches

def stratified_sample(y, n_samples):
    
    if torch.is_tensor(y):
        y = y.numpy()
    elif isinstance(y, list):
        y = np.array(y)

    if n_samples > 0:
        # Step 1: Get all the indices
        all_indices = np.arange(len(y))

        # Step 2: Sample indices proportionally to class weights
        unique_classes, counts_classes = np.unique(y, return_counts=True)
        class_weights = counts_classes / counts_classes.sum()  # Sums to 1

        # Step 3: Assign weights to each sample
        sample_weights = np.zeros(len(y))
        for c, w in zip(unique_classes, class_weights):
            sample_weights[y == c] = w

        # Normalize again to handle floating-point errors
        sample_weights = sample_weights / sample_weights.sum()

        # Step 4: Sample indices
        indices = np.random.choice(
            all_indices,
            size=n_samples,
            replace=False,
            p=sample_weights  # Guaranteed to sum to 1
        )
        return indices
    else:
        raise ValueError("n_samples must be greater than 0")


##########################################################
## LOSS AND METRICS UTILS
##########################################################


def soft_find_first_big_drop(x, alpha=2.0, beta=1.0):
    N = len(x)
    diffs = x[:-1].clone() - x[1:].clone() # Compute discrete differences
    abs_index = torch.argmax(diffs) + 1 
    # Compute position weights centered around the middle of the tensor
    center = (N - 1) / 2  # Midpoint
    positions = torch.arange(N - 1, dtype=x.dtype, device=x.device)
    position_weights = torch.exp(-beta * (positions - center) ** 2)

    # Compute softmax on negative differences, incorporating position bias
    weights = torch.softmax(-alpha * diffs * position_weights, dim=0)

    # Compute soft index as weighted sum of indices
    soft_index = (weights * positions).sum().type(torch.int) + 1

    return abs_index, soft_index  # Convert to integer index


def find_split_layer(nc_metrics, alpha=2.0, beta=3.0, verbose=False):
    split_layer = -1
    
    all_ranks = []
    
    for _, value in nc_metrics.items():
        for metric_name, metric_value in value.items():
            if metric_name == 'numerical_rank':
                metric_value = int(metric_value)
                all_ranks.append(metric_value)

    abs_split_layer, soft_split_layer = soft_find_first_big_drop(torch.tensor(all_ranks, dtype=torch.int), alpha=alpha, beta=beta)

    abs_split_layer = int(abs_split_layer)
    soft_split_layer = int(soft_split_layer)
    
    
    if verbose:
        print(f"Absolute split layer: {abs_split_layer}, Soft split layer: {soft_split_layer} for ranks: {all_ranks}")

    return abs_split_layer, soft_split_layer, np.array(all_ranks)


class CrossEntropyWithEntropyReg(torch.nn.CrossEntropyLoss):
    """
    CrossEntropyLoss with entropy regularization.
    """
    def __init__(self, alpha=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def entropy_reg(self, embeddings):
        """
        Computes entropy regularization term based on the pairwise distances between embeddings.
        """
        N = embeddings.size(0)
        dist_matrix = torch.cdist(embeddings, embeddings, p=2) \
                                + torch.eye(N, device=embeddings.device) * 1e6 \
                                    + torch.ones((N, N), device=embeddings.device) * 1e-6  # Avoid division by zero and diag
        # get the smallest distance for each point
        min_distances, _ = torch.min(dist_matrix, dim=1)
        return - self.alpha * torch.mean(torch.log(min_distances))

    def forward(self, outputs, embeddings, targets):
        ce_loss = super().forward(outputs, targets)
        if self.alpha == 0:
            return ce_loss
        else:
            return ce_loss + self.entropy_reg(embeddings)
        

# code taken from pypi package torch-scatter

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor,
                index: torch.Tensor,
                dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: torch.Tensor,
                 index: torch.Tensor,
                 dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out


def scatter_min(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor,
            index: torch.Tensor,
            dim: int = -1,
            out: Optional[torch.Tensor] = None,
            dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError
        

##########################################################
## TRAINER UTILS
##########################################################


class Layer:
    def __init__(self, layer_idx, patience=15):
        self.layer_idx = layer_idx
        self.starting_val = None
        self.current_val = None
        self.representative = True
        self.moving = False
        self.layer_patience = patience
        self.counter = 0
        self.changed = False
        

    def set_val(self, metric):
        if self.starting_val is None:
            self.starting_val = metric
        else:
            self.current_val = metric
            self.classify()

    def classify(self):
        if self.current_val is not None and self.starting_val is not None:
            if self.current_val > self.starting_val:
                if self.representative == False:
                    self.counter = 0
                    self.moving = True
                    self.changed = True
                else:
                    self.changed = False
                    self.counter += 1
                    if self.counter >= self.layer_patience:
                        self.moving = False

                self.representative = True
            else:
                if self.representative == True:
                    self.counter = 0
                    self.moving = True
                    self.changed = True
                else:
                    self.changed = False
                    self.counter += 1
                    if self.counter >= self.layer_patience:
                        self.moving = False
                self.representative = False
