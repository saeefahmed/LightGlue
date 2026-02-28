import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from lightglue.viz2d import plot_images, plot_matches
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load extractor & matcher
extractor = SuperPoint().eval().to(device)
matcher = LightGlue(features='superpoint').eval().to(device)

# Load images
image0 = load_image('assets/sacre_coeur1.jpg').to(device)
image1 = load_image('assets/sacre_coeur2.jpg').to(device)

# Extract features
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

# Match
matches01 = matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

# Get matched keypoints
kpts0 = feats0['keypoints']
kpts1 = feats1['keypoints']
matches = matches01['matches']

m_kpts0 = kpts0[matches[:, 0]]
m_kpts1 = kpts1[matches[:, 1]]

# Plot
plot_images([image0.cpu(), image1.cpu()])
plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.5)
plt.show()