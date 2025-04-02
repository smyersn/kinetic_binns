from modules.utils.imports import *

def create_data_histogram(training_data, device):
    u = training_data[:, 0].flatten()
    v = training_data[:, 1].flatten()

    hist = torchist.normalize(torchist.histogramdd(training_data, bins=10, 
                                low=[min(u).item(), min(v).item()], 
                                upp=[max(u).item(), max(v).item()]))[0]

    edges = torchist.histogramdd_edges(training_data, bins=10, 
                                    low=[min(u).item(), min(v).item()], 
                                    upp=[max(u).item(), max(v).item()])
    edges[0] = edges[0].to(device)
    edges[1] = edges[1].to(device)
    
    return hist, edges

def calc_density(uv, hist, edges):
    # uv_indices = torch.zeros((len(uv), len(uv.T)))
    # # iterate over columns
    # for i in range(len(uv.T)):
    #     buckets = torch.bucketize(uv[:, i], edges[i], right=True)
    #     # convert from buckets to indices, correct so max value falls in last bucket
    #     indices = torch.where(buckets != 0, buckets-1, buckets)
    #     indices_corrected = torch.where(indices == len(edges[i])-1, indices-1, indices)
    #     uv_indices[:, i] = indices_corrected
        
    # density = torch.tensor([hist[int(indices[0]), int(indices[1])] for indices in uv_indices]).to(uv.device)
    
    x_bin_indices = torch.searchsorted(edges[0].contiguous(), uv[:, 0].contiguous()) - 1
    y_bin_indices = torch.searchsorted(edges[1].contiguous(), uv[:, 1].contiguous()) - 1

    # Clip indices to be within the valid range
    x_bin_indices = torch.clamp(x_bin_indices, 0, hist.shape[1] - 1)
    y_bin_indices = torch.clamp(y_bin_indices, 0, hist.shape[0] - 1)

    # Step 2: Gather densities from the histogram using the bin indices
    density = hist[y_bin_indices.to('cpu'), x_bin_indices.to('cpu')]
                
    return density