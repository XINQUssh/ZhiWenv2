import os, glob, random, math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def load_img(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

base1 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/无贴屏'
base2 = 'f:/1111/指纹/dataRet_new_processed/dataRet_new_processed/不贴屏/不贴屏'

finger_imgs_train = {}
finger_imgs_test = {}
finger_labels = {}
fi = 0

for finger in sorted(os.listdir(base1)):
    rpath = os.path.join(base1, finger, 'Rgd1245')
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    all_imgs = []
    for p in imgs_paths:
        img = load_img(p).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        all_imgs.append(img)
    finger_imgs_train[finger] = all_imgs[:70]
    finger_imgs_test[finger] = all_imgs[70:]
    finger_labels[finger] = fi
    fi += 1

for finger in sorted(os.listdir(base2)):
    rpath = os.path.join(base2, finger, 'Rgd1237')
    if not os.path.exists(rpath):
        continue
    imgs_paths = sorted(glob.glob(os.path.join(rpath, '*.bmp')))
    all_imgs = []
    for p in imgs_paths:
        img = load_img(p).astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        all_imgs.append(img)
    finger_imgs_train[finger] = all_imgs[:70]
    finger_imgs_test[finger] = all_imgs[70:]
    finger_labels[finger] = fi
    fi += 1

n_classes = fi
fingers = list(finger_imgs_train.keys())
print(f"Total fingers: {n_classes}")

class ArcFaceLoss(nn.Module):
    def __init__(self, embed_dim, n_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, emb, labels):
        emb_norm = F.normalize(emb, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        cos_theta = torch.mm(emb_norm, w_norm.t()).clamp(-1, 1)
        theta = torch.acos(cos_theta)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = torch.cos(theta + one_hot * self.m) * self.s
        return F.cross_entropy(output, labels)

class TemplateDataset(Dataset):
    def __init__(self, finger_imgs, labels, n_template=30, n_samples=100, augment=True):
        self.finger_imgs = finger_imgs
        self.labels = labels
        self.n_template = n_template
        self.n_samples = n_samples
        self.augment = augment
        self.fingers = list(finger_imgs.keys())
        self.total = len(self.fingers) * n_samples

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        finger = self.fingers[idx // self.n_samples]
        imgs = self.finger_imgs[finger]
        chosen = random.sample(range(len(imgs)), min(self.n_template, len(imgs)))
        selected = [imgs[i].copy() for i in chosen]
        ref = selected[0].flatten()
        aligned = [selected[0]]
        for img in selected[1:]:
            corr = np.corrcoef(ref, img.flatten())[0, 1]
            aligned.append(-img if corr < 0 else img)
        template = np.mean(aligned, axis=0).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                template = -template
            template += np.random.randn(*template.shape).astype(np.float32) * 0.03
            dx, dy = random.randint(-2, 2), random.randint(-2, 2)
            template = np.roll(np.roll(template, dx, axis=1), dy, axis=0)
        return torch.tensor(template).unsqueeze(0), self.labels[finger]

class DeepCNN(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.15),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.embed = nn.Linear(512 * 4, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        emb = self.bn(self.embed(x))
        return emb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepCNN(embed_dim=256).to(device)
arcface = ArcFaceLoss(256, n_classes, s=30, m=0.5).to(device)
optimizer = optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)

train_ds = TemplateDataset(finger_imgs_train, finger_labels, n_template=40, n_samples=150, augment=True)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

print("Training ArcFace CNN...")
for epoch in range(120):
    model.train()
    arcface.train()
    total_loss, correct, total = 0, 0, 0
    for X_b, Y_b in train_loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        emb = model(X_b)
        loss = arcface(emb, Y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        emb_norm = F.normalize(emb, dim=1)
        w_norm = F.normalize(arcface.weight, dim=1)
        cos = torch.mm(emb_norm, w_norm.t())
        _, pred = cos.max(1)
        correct += pred.eq(Y_b).sum().item()
        total += Y_b.size(0)
    scheduler.step()
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, acc={correct/total*100:.1f}%")

# Test
model.eval()
print("\n=== ArcFace CNN Test Results ===")

def make_template(imgs, n=25):
    chosen = random.sample(range(len(imgs)), min(n, len(imgs)))
    selected = [imgs[i].copy() for i in chosen]
    ref = selected[0].flatten()
    aligned = [selected[0]]
    for img in selected[1:]:
        corr = np.corrcoef(ref, img.flatten())[0, 1]
        aligned.append(-img if corr < 0 else img)
    return np.mean(aligned, axis=0).astype(np.float32)

test_embs = {}
for finger in fingers:
    embs = []
    for _ in range(20):
        t = make_template(finger_imgs_test[finger], n=15)
        t = torch.tensor(t).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(t)
        emb = F.normalize(emb, dim=1).cpu().numpy().flatten()
        embs.append(emb)
    test_embs[finger] = embs

genuine, impostor = [], []
for finger in fingers:
    embs = test_embs[finger]
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            genuine.append(np.dot(embs[i], embs[j]))

for i in range(len(fingers)):
    for j in range(i + 1, len(fingers)):
        for a in test_embs[fingers[i]][:5]:
            for b in test_embs[fingers[j]][:5]:
                impostor.append(np.dot(a, b))

labels = [1] * len(genuine) + [0] * len(impostor)
scores = genuine + impostor
fpr, tpr, _ = roc_curve(labels, scores)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.abs(fnr - fpr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

print(f"Genuine: mean={np.mean(genuine):.4f}, std={np.std(genuine):.4f}, min={np.min(genuine):.4f}")
print(f"Impostor: mean={np.mean(impostor):.4f}, std={np.std(impostor):.4f}, max={np.max(impostor):.4f}")
print(f"EER: {eer*100:.2f}%")

for tfnr in [0.01, 0.03, 0.05, 0.10, 0.20]:
    idx = np.argmin(np.abs(fnr - tfnr))
    print(f"FFR={tfnr*100:.0f}% -> FAR={fpr[idx]*100:.4f}%")

target_far = 0.00002
idx_far = np.argmin(np.abs(fpr - target_far))
print(f"FAR=0.002% -> FFR={fnr[idx_far]*100:.2f}%")

gen_min = np.min(genuine)
imp_max = np.max(impostor)
print(f"\ngen_min={gen_min:.4f}, imp_max={imp_max:.4f}")
if gen_min > imp_max:
    print("*** PERFECT SEPARATION ***")
else:
    overlap_g = sum(1 for x in genuine if x < imp_max)
    overlap_i = sum(1 for x in impostor if x > gen_min)
    print(f"Overlap genuine: {overlap_g}/{len(genuine)}, impostor: {overlap_i}/{len(impostor)}")
