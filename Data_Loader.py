import numpy as np, torch
import scipy.io as sio
from torch.utils.data import Dataset
from torchvision import transforms 
from PIL import Image 

# mat = sio.loadmat('bandgap_data.mat')
# X_raw = mat['feature_raw'] # raw features
# X_shapefreq = mat['feature_shapefreq'] # shape frequency features
# dispersion = mat['dispersion'] # dispersion curves
# print('Shape of raw features', X_raw.shape)
# print('Shape of shape-frequency-features', X_shapefreq.shape)
# print('Shape of dispersion curves', dispersion.shape)


def get_labels_and_gaps(freq_centr, freq_gap, freq_range, size_min = 0.0):
    """Calculate whether a band gap exist in certain freq range"""
    labels = (freq_centr>0).astype('int32')
    freq_lower = freq_centr - 0.5 * freq_gap
    freq_upper = freq_centr + 0.5 * freq_gap
    labels_new = np.zeros((freq_centr.shape[0],freq_range.shape[0]))
    max_gaps = np.zeros((freq_centr.shape[0],freq_range.shape[0]))
    for i in range(labels_new.shape[1]):
        labels_new[:,i] = ((np.fmin(freq_upper,freq_range[i,1])-np.fmax(freq_lower,freq_range[i,0])>size_min).astype('int32').sum(1)>0).astype('int32')
        max_gaps[:,i] = (np.fmin(freq_upper,freq_range[i,1])-np.fmax(freq_lower,freq_range[i,0])).max(1)
    
    max_gaps = np.fmax(max_gaps,0)
    return labels_new

#Make Dataset Class for CNN
#This jelper takes out 15 feature input array and makes our 10x10 unit cell (Needed for the CNN to have a picture)
def expand_15_to_10x10(bits15):
    #place the 15 features into the upper triangle 
    tri = np.zeros((5,5), dtype = np.uint8)
    k = 0
    for i in range(5):
        for j in range(i, 5):
            tri[i, j] = bits15[k]
            k += 1
    #enforce symmetry: copy the upper triangle to the lower triangle 
    tri = tri | tri.T
    #enforce horiz/vert symmetry by tiling the 5x5 block 
    top = np.concatenate([tri, np.fliplr(tri)], axis=1)
    #stack that with its top-bottom flip
    grid10 = np.concatenate([top, np.flipud(top)], axis=0)
    return grid10

#Dataset class
class UnitCellDS(Dataset):
    def __init__(self, X15: np.ndarray, y: np.ndarray, resize=64):
        self.X15 = X15.astype(np.uint8)
        self.y = torch.from_numpy(y.astype(np.int64))
        self.itm = transforms.Compose([
            transforms.Resize(resize), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])

    #returns size of so dataloader knows how many samples exist 
    def __len__(self):
        return len(self.X15)
    #This will build a training sample on demand
    def __getitem__(self, i):
        grid10 = expand_15_to_10x10(self.X15[i]) * 255 #0/255 image 
        img = Image.fromarray(grid10.astype(np.uint8), mode="L")
        x = self.itm(img)
        y = self.y[i]
        return x, y


#Function to load arrays: 
def load_bandgap_data(mat_file='bandgap_data.mat', resize=64):
    mat = sio.loadmat(mat_file)
    X_raw = mat['feature_raw'] # raw features
    dispersion = mat['dispersion'] # dispersion curves

    # Calculate frequency gaps
    freq_gap_inplane = dispersion[:,1:,:].min(2)-dispersion[:,:dispersion.shape[1]-1,:].max(2)
    freq_centr_inplane = (dispersion[:,1:,:].min(2)+dispersion[:,:dispersion.shape[1]-1,:].max(2))/2
    freq_gap_inplane[np.where(freq_gap_inplane<0.01)] = 0
    freq_centr_inplane[np.where(freq_gap_inplane<0.01)] = 0

    freq_range_inplane = np.array([[0,1000],[1000,2000],[2000,3000], [3000, 4000], [4000, 5000]])
    labels = get_labels_and_gaps(freq_centr_inplane, freq_gap_inplane, freq_range_inplane, size_min = 0.0)

    dataset = UnitCellDS(X_raw, labels, resize=resize)

    return dataset



