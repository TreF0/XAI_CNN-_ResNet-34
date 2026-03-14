import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json

PROJECT_DIR = Path("folder path")
MODEL_PATH  = PROJECT_DIR / "model path"
CLASS_PATH  = PROJECT_DIR / "class path"
IMG_PATH    = PROJECT_DIR / "picture path"
OUT_DIR = MODEL_PATH.parent / "Analiza"
OUT_DIR.mkdir(exist_ok=True)
SAVE_PATH = OUT_DIR / "analiza_warstwowa.png"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def tensor_to_img(x):
    """[1,3,224,224] → uint8 HWC"""
    x = x.squeeze().permute(1, 2, 0).cpu().numpy()
    x = x * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients
        activations = self.activations

     
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        eps = 1e-8
        alpha_numer = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_activations * grads_power_3
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, eps)
        alpha = alpha_numer / alpha_denom
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3), keepdim=True)
        weighted_activations = weights * activations
        heatmap = weighted_activations.sum(dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy()

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        self.target_layer.register_forward_hook(forward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        _ = self.model(input_tensor)

        with torch.no_grad():
            acts = F.relu(self.activations)
            B, C, H, W = acts.shape
            up_size = input_tensor.shape[2:]

            cams_up = []
            scores = []

            for c in range(C):
                act = acts[:, c:c+1, :, :]
                act_up = F.interpolate(
                    act, size=up_size,
                    mode='bilinear', align_corners=False
                )

                a_min = act_up.min()
                a_max = act_up.max()
                if (a_max - a_min) < 1e-8:
                    continue

                act_norm = (act_up - a_min) / (a_max - a_min + 1e-8)

                masked_input = input_tensor * act_norm
                score = self.model(masked_input)[0, class_idx].item()

                scores.append(score)
                cams_up.append(act_norm.squeeze().cpu())

            if len(cams_up) == 0:
                return np.zeros(up_size, dtype=np.float32)

            scores = torch.tensor(scores, device=input_tensor.device, dtype=torch.float32)
            weights = F.softmax(scores, dim=0).view(-1, 1, 1)

            cams_stack = torch.stack(cams_up, dim=0)
            cam = torch.sum(weights * cams_stack, dim=0)

            cam = cam.clamp(min=0)
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)

            return cam.cpu().numpy()

class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()
        def bwd_hook(module, gin, gout):
            self.gradients = gout[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        A = self.activations
        G = self.gradients

        cam = F.relu((A * G).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy().astype(np.float32)
        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_PATH) as f:
    class_to_idx = json.load(f)
num_classes = len(class_to_idx)
idx_to_class = {v: k for k, v in class_to_idx.items()}

model = models.resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

img = transform(Image.open(IMG_PATH).convert("RGB"))
img = img.unsqueeze(0).to(device)
orig_uint8 = tensor_to_img(img)

pred = model(img).argmax().item()

layers = [
    ("layer1", model.layer1[-1].conv2),
    ("layer2", model.layer2[-1].conv2),
    ("layer3", model.layer3[-1].conv2),
    ("layer4", model.layer4[-1].conv2),
]

fig, ax = plt.subplots(3, 5, figsize=(22, 12))

for r in range(3):
    ax[r, 0].imshow(orig_uint8[..., ::-1])
    ax[r, 0].set_title("Oryginał")
    ax[r, 0].axis("off")

H, W = 224, 224

for c, (name, layer) in enumerate(layers, start=1):
    gc = GradCAM(model, layer)
    cam = gc.generate(img, pred)
    cam = cv2.resize(cam.astype(np.float32), (W, H))
    ax[0, c].imshow(cam, cmap="jet")
    ax[0, c].set_title(f"Grad-CAM {name}")
    ax[0, c].axis("off")

for c, (name, layer) in enumerate(layers, start=1):
    sc = ScoreCAM(model, layer)
    cam = sc.generate(img, pred)
    cam = cv2.resize(cam.astype(np.float32), (W, H))
    ax[1, c].imshow(cam, cmap="jet")
    ax[1, c].set_title(f"Score-CAM {name}")
    ax[1, c].axis("off")

for i, (name, layer) in enumerate(layers):
    if name == "layer4":
        ax[2, 4].imshow(np.zeros((H, W)))
        ax[2, 4].set_title("Layer-CAM layer4 (brak)")
        ax[2, 4].axis("off")
        continue

    lc = LayerCAM(model, layer)
    cam = lc.generate(img, pred)
    cam = cv2.resize(cam.astype(np.float32), (W, H))
    ax[2, i + 1].imshow(cam, cmap="jet")
    ax[2, i + 1].set_title(f"Layer-CAM {name}")
    ax[2, i + 1].axis("off")

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=200, bbox_inches="tight")
plt.show()
plt.close()

print("Zapisano:", SAVE_PATH)
