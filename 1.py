from torchvision import models, transforms, datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import os
import time
import cv2
import numpy as np
from datetime import datetime
import math
from glob import glob
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF
import random


class RandomizeLeafBackground:
    def __call__(self, img):
        arr = np.array(img)
        mask = (arr != [0, 0, 0]).any(axis=-1)
        # losowy kolor
        rand_color = np.random.randint(0, 256, 3)
        arr[~mask] = rand_color
        return Image.fromarray(arr.astype(np.uint8))

class ReplaceWhiteWithBlack:
    def __call__(self, img):
        img = img.convert("RGB")
        np_img = np.array(img)
        mask = (np_img >= 240).all(axis=2)  # białe piksele (lub bardzo jasne)
        np_img[mask] = [0, 0, 0]  # zamień na czarne
        return Image.fromarray(np_img)
#nowy inny loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma


    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return hard_example_loss(focal, retain=0.6)

class MaskCircle:
    def __init__(self, fill=(0,0,0)):
        self.fill = fill
    def __call__(self, img):
        arr = np.array(img)
        h, w = arr.shape[:2]
        center = (w // 2, h // 2)
        radius = min(h, w) // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        mask = dist <= radius
        out = np.zeros_like(arr)
        out[:,:,:] = self.fill
        out[mask] = arr[mask]
        return Image.fromarray(out)


#dane konfiguracji sieci
typ = True #true - lisc; false - skora
max_epoch = 60
batch_size_= 256
num_workers_= 4
lr_=3e-4
weight_decay_=1e-4
T_max_= max_epoch

if typ:
    data_root = "/home/praktyki/Pulpit/23_07/baza_lisc_duza"
    base_results_dir = "/home/praktyki/Pulpit/23_07/zapis_lisc"
    specific_transforms = [
        #ReplaceWhiteWithBlack(),
        #RandomizeLeafBackground(),
        
    ]
else:
    data_root = "/home/praktyki/Pulpit/23_07/podzial_skora"
    base_results_dir = "/home/praktyki/Pulpit/23_07/zapis_skora"
    specific_transforms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

# Ogólne transformacje
common_transforms = [
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

# Składanie pełnej transformacji
trans = transforms.Compose(specific_transforms + common_transforms)

# Datasety
train_ds = datasets.ImageFolder(os.path.join(data_root, "train"), trans)
val_ds   = datasets.ImageFolder(os.path.join(data_root, "val"), trans)
test_ds  = datasets.ImageFolder(os.path.join(data_root, "test"), trans)

# Tworzenie folderów zapisu
os.makedirs(base_results_dir, exist_ok=True)
existing = [int(os.path.basename(f)) for f in glob(os.path.join(base_results_dir, '*')) if os.path.basename(f).isdigit()]
next_id = max(existing) + 1 if existing else 1
save_path = os.path.join(base_results_dir, str(next_id))
os.makedirs(save_path, exist_ok=True)

plots_path = os.path.join(save_path, "Wykresy")
os.makedirs(plots_path, exist_ok=True)

timestamp_str = datetime.now().strftime("%H-%M-%S_%Y-%m-%d")
open(os.path.join(save_path, f"{timestamp_str}.txt"), "w").close()

print(f"Folder zapisu: {save_path}")
print(f"Utworzono plik znacznikowy: {timestamp_str}.txt")

# Loadery danych
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size_, shuffle=True, num_workers=num_workers_)
test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=batch_size_, shuffle=True, num_workers=num_workers_)
val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size_, shuffle=False, num_workers=num_workers_)

# Urządzenie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, len(train_ds.classes)) 
model.to(device)

# Optymalizator i scheduler
opt = torch.optim.AdamW(model.parameters(), lr=lr_, weight_decay=weight_decay_)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max_)

val_accuracy_tab = []

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

def hard_example_loss(loss_tensor, retain=0.7):  # 70% najtrudniejszych
    k = int(retain * loss_tensor.size(0))
    topk_vals, _ = torch.topk(loss_tensor, k)
    return topk_vals.mean()


# ==== Inicjalizacja zmiennych ====
train_losses = []
val_accuracy_tab = []
reference_loss = None
stable_counter = 0
stop_training = False
patience = 20
delta = 0.000001  # potrzebne dla early stopping

loss_fn = FocalLoss(gamma=2.0)

# ==== Trenowanie modelu ====
for epoch in range(max_epoch):
    if stop_training:
        break

    model.train()
    running_loss = 0
    batch_count = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        opt.step()

        current_loss = loss.item()
        running_loss += current_loss
        batch_count += 1

        # Early stop
        if reference_loss is None:
            reference_loss = current_loss
            stable_counter = 0
        elif abs(current_loss - reference_loss) <= delta:
            stable_counter += 1
            if stable_counter >= patience:
                print(f"EARLY STOPPING: zatrzymano po {stable_counter} stabilnych iteracjach. Loss: {current_loss:.6f}")
                stop_training = True
                break
        else:
            reference_loss = current_loss
            stable_counter = 0

    # Średnia strata z epoki
    avg_loss = running_loss / batch_count
    train_losses.append(avg_loss)

    # Walidacja po każdej epoce
    val_acc = evaluate(model, val_loader, device)
    val_accuracy_tab.append(val_acc)

    print(f"Epoka {epoch+1}/{max_epoch} | Avg Loss: {avg_loss:.6f} | Val Acc: {val_acc:.2f}%")
    sched.step()

# --- Testowanie na zbiorze testowym ---
model.eval()
final_preds, final_targets = [], []
correct, total = 0, 0
accuracy_tab = []
class_names = train_ds.classes
num_classes = len(class_names)

with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        out_test = model(x_test)
        preds = torch.argmax(out_test, dim=1)
        correct += (preds == y_test).sum().item()
        total += y_test.size(0)
        final_preds.extend(preds.cpu().numpy())
        final_targets.extend(y_test.cpu().numpy())

acc = 100 * correct / total
accuracy_tab.append(acc)
print(f"Końcowy test: Dokładność = {acc:.2f}%")

# --- Macierz pomyłek ---
cm = confusion_matrix(final_targets, final_preds)

# --- t-SNE ---
features, labels = [], []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        feats = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(x_batch))))))))
        feats = feats.view(feats.size(0), -1)
        features.append(feats.cpu().numpy())
        labels.extend(y_batch.numpy())
features = np.concatenate(features)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(features)

# --- Czyste obrazy ---
clean_images, images_per_class = {}, {}
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = torch.argmax(model(x_batch), dim=1)
        for img_tensor, true_label, pred_label in zip(x_batch, y_batch, preds):
            if true_label == pred_label:
                class_name = class_names[true_label.item()]
                if class_name not in clean_images:
                    clean_images[class_name] = img_tensor.cpu()
                if class_name not in images_per_class:
                    images_per_class[class_name] = (img_tensor.cpu(), true_label.item())
        if len(images_per_class) == num_classes:
            break

# --- Rysowanie czystych obrazów ---
fig, axes = plt.subplots(math.ceil(len(clean_images)/4), 4, figsize=(12, 3 * math.ceil(len(clean_images)/4)))
fig.suptitle("Czyste obrazy (poprawnie sklasyfikowane)", fontsize=18)
for ax, (class_name, img_tensor) in zip(axes.flat, clean_images.items()):
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(class_name, fontsize=8)
    ax.axis("off")
for i in range(len(clean_images), len(axes.flat)):
    axes.flat[i].axis("off")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig(os.path.join(plots_path, "clean_class_examples.png"))
plt.close(fig)

# --- Grad-CAM funkcja pomocnicza ---
def generate_gradcam(model, image_tensor, class_idx, target_layer):
    gradients, activations = [], []
    def fw_hook(module, input, output): activations.append(output)
    def bw_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
    handle_fw = target_layer.register_forward_hook(fw_hook)
    handle_bw = target_layer.register_full_backward_hook(bw_hook)
    with torch.enable_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        image_tensor.requires_grad = True
        output = model(image_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        model.zero_grad()
        output.backward(gradient=one_hot)
        grads, acts = gradients[0].squeeze(0), activations[0].squeeze(0)
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(0)).detach().cpu().numpy()
        cam = cv2.resize((cam - cam.min()) / (np.ptp(cam) + 1e-5), (224, 224))
        image = image_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.uint8(255 * image), 0.5, heatmap, 0.5, 0)
    handle_fw.remove(); handle_bw.remove()
    return overlay, output.argmax().item()

# --- Grad-CAM siatka (wiele warstw) ---
layers_to_inspect = {
    "conv1": model.conv1,
    "layer1": model.layer1[0].conv2,
    "layer2": model.layer2[0].conv2,
    "layer3": model.layer3[0].conv2,
    "layer4": model.layer4[2].conv2
}
for layer_name, layer in layers_to_inspect.items():
    print(f"Grad-CAM: {layer_name}")
    fig, axes = plt.subplots(math.ceil(num_classes/4), 4, figsize=(16, 3 * math.ceil(num_classes/4)))
    fig.suptitle(f"Grad-CAM – {layer_name}", fontsize=16)
    for ax, (cls_name, (img, label)) in zip(axes.flat, images_per_class.items()):
        grad_img, pred = generate_gradcam(model, img, label, layer)
        ax.imshow(grad_img[..., ::-1])
        ax.set_title(f"{cls_name}\nPred: {class_names[pred]}", fontsize=8)
        ax.axis("off")
    for i in range(len(images_per_class), len(axes.flat)):
        axes.flat[i].axis("off")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = os.path.join(plots_path, f"gradcam_{layer_name}.png")
    fig.savefig(fname)
    plt.close(fig)

# --- Zbiorczy wykres diagnostyczny ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0, 0].plot(train_losses); axes[0, 0].set_title("Train Loss"); axes[0, 0].grid(True)
axes[0, 1].plot(val_accuracy_tab); axes[0, 1].set_title("Test Accuracy"); axes[0, 1].grid(True)
for i, name in enumerate(class_names):
    idxs = np.array(labels) == i
    axes[0, 2].scatter(features_2d[idxs, 0], features_2d[idxs, 1], label=name, alpha=0.6)
axes[0, 2].set_title("t-SNE"); axes[0, 2].legend(fontsize=7); axes[0, 2].grid(True)
axes[1, 0].hist(model.conv1.weight.data.cpu().numpy().flatten(), bins=50, alpha=0.75)
axes[1, 0].set_title("Histogram wag: conv1")
axes[1, 1].hist(model.fc.weight.data.cpu().numpy().flatten(), bins=50, alpha=0.75)
axes[1, 1].set_title("Histogram wag: fc")
axes[1, 2].axis('off')
fig.tight_layout()
fig.savefig(os.path.join(plots_path, "diagnostics_metrics.png"))
plt.close(fig)

# --- Macierz pomyłek ---
fig, ax = plt.subplots(figsize=(12, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title("Macierz pomyłek")
plt.tight_layout()
fig.savefig(os.path.join(plots_path, "confusion_matrix.png"))
plt.close(fig)
print("Zapisano wszystkie wykresy i diagnostykę")

# Zapis metryk, konfiguracji i modelu
training_metrics = {
    "train_loss": train_losses,
    "val_accuracy": val_accuracy_tab,
    "test_accuracy": accuracy_tab
}
with open(os.path.join(save_path, "training_metrics.json"), "w") as f:
    json.dump(training_metrics, f, indent=4)
print("Zapisano metryki do 'training_metrics.json'.")

training_config = {
    "max_epoch": max_epoch,
    "batch_size": batch_size_,
    "learning_rate": lr_,
    "weight_decay": weight_decay_,
    "scheduler": "CosineAnnealingLR",
    "T_max": T_max_,
    "patience": patience,
    "min_delta": delta,
    "model": "resnet34",
    "transform": {
        "resize": 256,
        "crop": 224,
        "horizontal_flip": False,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225]
    },
    "num_classes": len(train_ds.classes),
    "device": str(device)
}
with open(os.path.join(save_path, "training_config.json"), "w") as f:
    json.dump(training_config, f, indent=4)
print("Zapisano konfigurację treningu do 'training_config.json'.")

# Zapis mapowania klas
with open(os.path.join(save_path, "class_index.json"), "w") as f:
    json.dump(train_ds.class_to_idx, f)
print("Zapisano klasy do 'class_index.json'.")

# Zapis modelu
torch.save(model.state_dict(), os.path.join(save_path, "resnet34_last.pth"))
print("Model zapisany jako 'resnet34_last.pth'.")