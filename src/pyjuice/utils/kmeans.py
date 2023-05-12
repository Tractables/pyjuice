import numpy as np
import torch

from typing import Union


def faiss_kmeans(data: Union[torch.Tensor,np.ndarray], n_clusters: int, niter: int = 100, nredo: int = 100):

    # Only import faiss if we run this function
    import faiss

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
        convert_to_torch = True
    else:
        convert_to_torch = False

    kmeans = faiss.Clustering(data.shape[1], n_clusters)
    kmeans.verbose = False
    kmeans.niter = niter
    kmeans.nredo = nredo
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    cfg.device = self.device_id
    index = faiss.GpuIndexFlatL2(
        faiss.StandardGpuResources(),
        data.shape[1],
        cfg
    )
    kmeans.train(data, index)
    centroids = faiss.vector_float_to_array(kmeans.centroids).reshape(n_clusters, data.shape[1])

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(np.ascontiguousarray(centroids))
    _, labels = index.search(data, 1)
    labels = labels.ravel()

    if convert_to_torch:
        labels = torch.from_numpy(labels)

    return labels