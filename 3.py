import torch
from torchvision import models, transforms
from PIL import Image
import json
import os
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
from torchvision.models.resnet import BasicBlock
import math
from torch.nn.utils import fuse_conv_bn_eval
from pathlib import Path
import random
from captum.attr import DeepLiftShap, LayerDeepLiftShap
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum")
import time
import csv
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import seaborn as sns



PROJECT_DIR = Path("folder path")

MODEL_PATH  = PROJECT_DIR / "model path"
CLASS_PATH  = PROJECT_DIR / "Class path"
VAL_ROOT    = PROJECT_DIR / "val path"
typ = False  # True - lisc, false - skora

OUT_DIR = MODEL_PATH.parent / "Analiza"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ile poprawnych obrazów z różnych klas analizować (domyślnie 1)
num_samples = 1               
max_baselines = 10           

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


LRP_     = True     # włącz/wyłącz
GradCAM_ = True     # włącz/wyłącz
DeepSHAP_= True     # włącz/wyłącz


with open(CLASS_PATH) as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes  = len(class_to_idx)


def disable_inplace_relu(m):
    if isinstance(m, nn.ReLU):
        m.inplace = False

def save_plama(orig_bgr, map01, method_name, save_dir, percentile=80):
    M = map01.astype(np.float32)
    thr = np.percentile(M, percentile)
    mask = (M >= thr).astype(np.uint8)

    overlay = orig_bgr.copy()
    overlay[mask == 1] = [255, 0, 0]  # plama = niebieski

    cv2.imwrite(os.path.join(save_dir, f"{method_name}_plama.png"), overlay)



def safe_basicblock_forward(self, x):
    identity = x
    if self.downsample is not None:
        identity = self.downsample(x)

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out, inplace=False)

    out = self.conv2(out)
    out = self.bn2(out)

    out = out + identity
    out = F.relu(out, inplace=False)
    return out


BasicBlock.forward = safe_basicblock_forward

# Główny model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet34(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_to_idx))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.apply(disable_inplace_relu)
model.to(device)

#PRZETWARZANIE WSZYSTKICH OBRAZKÓW (statystyki / t-SNE / CM)
total = 0
correct = 0
features_list = []
labels_list = []
wrong_images = []
wrong_preds = []
wrong_labels = []
y_true = []
y_pred = []

feature_extractor = create_feature_extractor(model, return_nodes={"avgpool": "features"})

for class_name in os.listdir(VAL_ROOT):
    class_dir = os.path.join(VAL_ROOT, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in os.listdir(class_dir):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp')):
            continue
        img_path = os.path.join(class_dir, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = feature_extractor(input_tensor)["features"].squeeze()
                output = model(input_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                pred_class = idx_to_class[pred_idx]
            features_list.append(feats.view(-1).cpu().numpy())
            labels_list.append(class_to_idx[class_name])
            total += 1
            if pred_class == class_name:
                correct += 1
            else:
                wrong_images.append(img)
                wrong_preds.append(pred_class)
                wrong_labels.append(class_name)
            y_true.append(class_to_idx[class_name])
            y_pred.append(pred_idx)
        except Exception as e:
            print(f"Błąd przy {img_path}: {e}")


def fuse_resnet_bn(m):
    for name, module in m.named_children():
        if isinstance(module, nn.Sequential):
            fuse_resnet_bn(module)
        elif isinstance(module, BasicBlock):
            if isinstance(module.conv1, nn.Conv2d) and isinstance(module.bn1, nn.BatchNorm2d):
                module.conv1 = fuse_conv_bn_eval(module.conv1, module.bn1)
                module.bn1 = nn.Identity()
            if isinstance(module.conv2, nn.Conv2d) and isinstance(module.bn2, nn.BatchNorm2d):
                module.conv2 = fuse_conv_bn_eval(module.conv2, module.bn2)
                module.bn2 = nn.Identity()
            if module.downsample is not None:
                layers = []
                conv = None
                for sub in module.downsample:
                    if isinstance(sub, nn.Conv2d):
                        conv = sub
                    elif isinstance(sub, nn.BatchNorm2d):
                        bn = sub
                        conv = fuse_conv_bn_eval(conv, bn)
                        layers.append(conv)
                module.downsample = nn.Sequential(*layers)
        else:
            fuse_resnet_bn(module)


model_lrp = models.resnet34(weights=None)
model_lrp.fc = torch.nn.Linear(model_lrp.fc.in_features, len(class_to_idx))
model_lrp.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model_lrp.eval()
model_lrp.apply(disable_inplace_relu)
model_lrp.to(device)
fuse_resnet_bn(model_lrp)

# LRP KLASA
class LRPResNet34(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.layer_list = [
            ('conv1', self.model.conv1),
            ('bn1', self.model.bn1),
            ('relu', self.model.relu),
            ('maxpool', self.model.maxpool),
            ('layer1', self.model.layer1),
            ('layer2', self.model.layer2),
            ('layer3', self.model.layer3),
            ('layer4', self.model.layer4),
            ('avgpool', self.model.avgpool),
            ('flatten', nn.Flatten()),
            ('fc', self.model.fc)
        ]

    def forward(self, x):
        acts = []
        for name, layer in self.layer_list:
            x = layer(x)
            acts.append((name, x))
        return acts

    def lrp(self, x, class_idx, eps=1e-6, rule="epsilon", gamma=0.25, alpha=1.0, beta=0.0, return_all_layers=False):
        x = x.clone().detach().to(self.device).requires_grad_(False)
        acts = [x]
        tmp = x
        for name, layer in self.layer_list:
            tmp = layer(tmp)
            acts.append(tmp)
        one_hot = torch.zeros_like(acts[-1])
        one_hot[0, class_idx] = acts[-1][0, class_idx]
        relevance = one_hot

        rel_maps = {}
        for i in reversed(range(len(self.layer_list))):
            name, layer = self.layer_list[i]
            inp = acts[i]
            out = acts[i+1]
            relevance = self._relprop(layer, inp, out, relevance, eps, rule, gamma, alpha, beta)
            if name in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']:
                rel_maps[name] = relevance.clone().detach()
        return rel_maps if return_all_layers else relevance

    def _relprop(self, layer, input_act, output_act, relevance, eps, rule, gamma, alpha, beta):
        if relevance is None:
            return torch.zeros_like(input_act)
        if isinstance(layer, nn.Linear):
            w = layer.weight.data
            b = None
            z = F.linear(input_act, w, b)
            if rule == "epsilon":
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance / z
                c = torch.matmul(s, w)
                return input_act * c
            elif rule == "gamma":
                w_gamma = w + gamma * torch.clamp(w, min=0)
                z = F.linear(input_act, w_gamma)
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance / z
                c = torch.matmul(s, w_gamma)
                return input_act * c
            elif rule == "alphabeta":
                w_pos = torch.clamp(w, min=0)
                w_neg = torch.clamp(w, max=0)
                z_pos = F.linear(input_act, w_pos)
                z_neg = F.linear(input_act, w_neg)
                z_pos += eps
                z_neg -= eps
                s_pos = relevance / z_pos
                s_neg = relevance / z_neg
                c = alpha * torch.matmul(s_pos, w_pos) + beta * torch.matmul(s_neg, w_neg)
                return input_act * c

        elif isinstance(layer, nn.Conv2d):
            w = layer.weight
            stride, padding, kernel_size = layer.stride, layer.padding, layer.kernel_size
            B, C_in, H, W = input_act.shape
            C_out = w.shape[0]
            x_unfold = F.unfold(input_act, kernel_size=kernel_size, stride=stride, padding=padding)
            w_flat = w.view(C_out, -1)

            if rule == "epsilon":
                z = torch.einsum("oc,bcl->bol", w_flat, x_unfold)
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance.view(B, C_out, -1) / z
                c = torch.einsum("oc,bol->bcl", w_flat, s)
            elif rule == "gamma":
                w_gamma = w + gamma * torch.clamp(w, min=0)
                w_flat = w_gamma.view(C_out, -1)
                z = torch.einsum("oc,bcl->bol", w_flat, x_unfold)
                z += eps * torch.where(z >= 0, 1., -1.)
                s = relevance.view(B, C_out, -1) / z
                c = torch.einsum("oc,bol->bcl", w_flat, s)
            elif rule == "alphabeta":
                w_pos = torch.clamp(w, min=0)
                w_neg = torch.clamp(w, max=0)
                w_pos_flat = w_pos.view(C_out, -1)
                w_neg_flat = w_neg.view(C_out, -1)
                z_pos = torch.einsum("oc,bcl->bol", w_pos_flat, x_unfold) + eps
                z_neg = torch.einsum("oc,bcl->bol", w_neg_flat, x_unfold) - eps
                R = relevance.view(B, C_out, -1)
                s_pos = R / z_pos
                s_neg = R / z_neg
                c_pos = torch.einsum("oc,bol->bcl", w_pos_flat, s_pos)
                c_neg = torch.einsum("oc,bol->bcl", w_neg_flat, s_neg)
                c = alpha * c_pos + beta * c_neg

            R_in = F.fold(c, output_size=(H, W), kernel_size=kernel_size, stride=stride, padding=padding)
            return input_act * R_in

        elif isinstance(layer, (nn.BatchNorm2d, nn.ReLU)):
            return relevance
        elif isinstance(layer, nn.MaxPool2d):
            return F.interpolate(relevance, size=input_act.shape[2:], mode="nearest")
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            return F.interpolate(relevance, size=input_act.shape[2:], mode="nearest")
        elif isinstance(layer, nn.Flatten):
            return relevance.view(input_act.shape)
        elif isinstance(layer, nn.Sequential):
            xs = [input_act]
            x = input_act
            for sublayer in layer:
                x = sublayer(x)
                xs.append(x)
            for i in reversed(range(len(layer))):
                sub_rel = self._relprop(layer[i], xs[i], xs[i+1], relevance, eps, rule, gamma, alpha, beta)
                if sub_rel is not None:
                    relevance = sub_rel
            return relevance
        elif isinstance(layer, BasicBlock):
            x = input_act
            if layer.downsample is not None:
                skip = layer.downsample(x)
            else:
                skip = x
            m1 = layer.conv1(x)
            m2 = layer.bn1(m1)
            m3 = layer.relu(m2)
            m4 = layer.conv2(m3)
            main = layer.bn2(m4)
            out = main + skip
            out = layer.relu(out)
            Z_main = main.abs()
            Z_skip = skip.abs()
            Z_sum = Z_main + Z_skip + 1e-12
            R_main = relevance * (Z_main / Z_sum)
            R_skip = relevance * (Z_skip / Z_sum)
            if layer.downsample is not None:
                R_skip = self._relprop(layer.downsample, x, skip, R_skip, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.bn2, m4, main, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.conv2, m3, m4, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.relu, m2, m3, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.bn1, m1, m2, R_main, eps, rule, gamma, alpha, beta)
            R_main = self._relprop(layer.conv1, x, m1, R_main, eps, rule, gamma, alpha, beta)
            return R_main + R_skip
        return relevance
# CAM KLASY 
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

class WeightedMultiLayerCAM:

    def __init__(self, model, target_layers, weights=None):

        self.model = model
        self.target_layers = target_layers

        # domyślne wagi 
        if weights is None:
            self.weights = {
                "layer1": 0.6,
                "layer2": 0.4,
                "layer3": 0,
            }
        else:
            self.weights = weights

        self.activations = {name: None for name in target_layers}
        self.gradients   = {name: None for name in target_layers}

        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.target_layers.items():

            def fwd_hook(module, inp, out, name=name):
                self.activations[name] = out.detach()

            def bwd_hook(module, gin, gout, name=name):
                self.gradients[name] = gout[0].detach()

            layer.register_forward_hook(fwd_hook)
            layer.register_full_backward_hook(bwd_hook)

    def _compute_single_cam(self, A, G):
        cam = F.relu((A * G).sum(dim=1).squeeze(0))  # [h,w]
        cam = cam.cpu().numpy().astype(np.float32)

        # jeśli mapa pusta → zwróć zero
        if cam.max() < 1e-10:
            return None

        # normalizacja 0–1
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam

    def generate(self, input_tensor, class_idx, img_size=(224, 224), sigma=3):
        # forward
        self.model.zero_grad()
        output = self.model(input_tensor)

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        cams = []

        for name in ["layer1", "layer2", "layer3"]:

            A = self.activations[name]
            G = self.gradients[name]

            cam = self._compute_single_cam(A, G)
            if cam is None:
                continue

            # skalowanie do wielkości obrazu
            cam = cv2.resize(
                cam,
                img_size,
                interpolation=cv2.INTER_LINEAR
            )

            # zapis z wagą
            cams.append(self.weights[name] * cam)

        if len(cams) == 0:
            return np.zeros(img_size, dtype=np.float32)

        # fuzja
        cam_final = np.sum(np.stack(cams, axis=0), axis=0)

        # normalizacja końcowa
        cam_final -= cam_final.min()
        cam_final /= cam_final.max() + 1e-8

        # opcjonalne wygładzenie
        if sigma > 0:
            cam_final = cv2.GaussianBlur(cam_final, (0, 0), sigma)

            cam_final -= cam_final.min()
            cam_final /= cam_final.max() + 1e-8

        return cam_final


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
        # forward, żeby mieć aktywacje
        self.model.zero_grad()
        _ = self.model(input_tensor)

        with torch.no_grad():
            # tylko dodatnie aktywacje
            acts = F.relu(self.activations)          # [1, C, H, W]
            B, C, H, W = acts.shape
            up_size = input_tensor.shape[2:]         # (H_in, W_in)

            cams_up = []   # lista masek [H_in, W_in]
            scores  = []   # wagi

            for c in range(C):
                act = acts[:, c:c+1, :, :]  # [1,1,H,W]

                # podniesienie do rozdzielczości wejścia
                act_up = F.interpolate(
                    act, size=up_size,
                    mode='bilinear', align_corners=False
                )  # [1,1,H_in,W_in]

                a_min = act_up.min()
                a_max = act_up.max()
                if (a_max - a_min) < 1e-8:
            
                    continue

                # normalizacja pojedynczej mapy do [0,1]
                act_norm = (act_up - a_min) / (a_max - a_min + 1e-8)

                # maskowanie wejścia
                masked_input = input_tensor * act_norm
                score = self.model(masked_input)[0, class_idx].item()

                scores.append(score)
                cams_up.append(act_norm.squeeze().cpu())   

            if len(cams_up) == 0:
                # awaryjnie zwróć zerową mapę
                return np.zeros(up_size, dtype=np.float32)

            scores = torch.tensor(
                scores, device=input_tensor.device, dtype=torch.float32
            )
            weights = F.softmax(scores, dim=0).view(-1, 1, 1)  # [K,1,1]

            cams_stack = torch.stack(cams_up, dim=0)  # [K, H_in, W_in]
            cam = torch.sum(weights * cams_stack, dim=0)       # [H_in, W_in]

            cam = cam.clamp(min=0)
            cam -= cam.min()
            cam /= (cam.max() + 1e-8)

            return cam.cpu().numpy()


# POMOCNICZE
def get_real_image_baselines(val_root, transform, num_baselines=10):
    baselines = []
    all_imgs = []
    for cls in os.listdir(val_root):
        cls_dir = os.path.join(val_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.png', '.tif', '.bmp')):
                all_imgs.append(os.path.join(cls_dir, fname))
    chosen_imgs = random.sample(all_imgs, min(num_baselines, len(all_imgs)))
    for img_path in chosen_imgs:
        img = Image.open(img_path).convert('RGB')
        baselines.append(transform(img))
    baselines = torch.stack(baselines)
    return baselines

def tensor_to_uint8_rgb(input_tensor):
    orig = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    orig = (orig * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    orig = np.clip(orig, 0, 1)
    return (orig * 255).astype(np.uint8)

# FUNKCJE POMOCNICZE DO NORMALIZACJI I METRYK

def normalize_map(arr, signed=False, out_size=(224, 224)):

    arr = np.array(arr, dtype=np.float32)

    if signed:
        v = np.abs(arr)
        vmax = np.percentile(v, 99) if np.any(v > 0) else 1.0
        arr = np.clip(arr, -vmax, vmax) / (vmax + 1e-8)   # [-1,1]
    else:
        mn = arr.min()
        mx = arr.max()
        if mx - mn < 1e-12:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - mn) / (mx - mn + 1e-8)           # [0,1]

    if out_size is not None:
        arr = cv2.resize(arr, out_size, interpolation=cv2.INTER_LINEAR)
    return arr


def normalize_to_01(arr, signed=False, out_size=(224,224)):

    a = np.array(arr, dtype=np.float32)
    if signed:
        v = np.abs(a)
        vmax = np.percentile(v, 99) if np.any(v > 0) else 1.0
        a = np.clip(a, -vmax, vmax) / (vmax + 1e-8)
        unit = (a + 1.0) / 2.0
    else:
        mn = a.min()
        mx = a.max()
        if mx - mn < 1e-12:
            unit = np.zeros_like(a, dtype=np.float32)
        else:
            unit = (a - mn) / (mx - mn + 1e-8)
    if out_size is not None:
        unit = cv2.resize(unit, out_size, interpolation=cv2.INTER_LINEAR)
    return unit


def compute_entropy(hm):
    p = np.abs(hm).astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def energy_concentration(hm, top_percent=5.0):
    v = np.abs(hm).flatten()
    if v.size == 0 or np.all(v == 0):
        return 0.0
    total = v.sum()
    k = max(1, int(len(v) * (top_percent / 100.0)))
    idx = np.argpartition(-v, k-1)[:k]
    return float(v[idx].sum() / (total + 1e-8))


def pearson_corr(a, b):
    x = a.flatten().astype(np.float64)
    y = b.flatten().astype(np.float64)
    if x.size != y.size:
        raise ValueError("Różne rozmiary map w pearson_corr")
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2)) + 1e-12
    return float(num / den)


def cosine_sim(a, b):

    x = a.flatten().astype(np.float64)
    y = b.flatten().astype(np.float64)
    num = np.dot(x, y)
    den = np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
    return float(num / den)


def ssim_map(a, b, signed_a=False, signed_b=False):

    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if signed_a:
        a = (a + 1.0) / 2.0
    if signed_b:
        b = (b + 1.0) / 2.0
    # Zakres danych [0,1]
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return float(ssim(a, b, data_range=1.0))


def occlusion_drop(input_tensor, heatmap, model, class_idx, device, k_frac=0.1):

    hm = np.abs(heatmap).astype(np.float32)
    flat = hm.flatten()
    n_pix = flat.size
    k = max(1, int(n_pix * k_frac))
    # indeksy najważniejszych pikseli
    idx_sorted = np.argpartition(-flat, k-1)[:k]
    mask = np.ones_like(flat, dtype=np.float32)
    mask[idx_sorted] = 0.0
    mask = mask.reshape(1, 1, 224, 224)  # broadcast na kanały

    mask_t = torch.from_numpy(mask).to(device)
    x = input_tensor.clone().to(device)
    x_occluded = x * mask_t  # wyzerowanie najważniejszych pikseli

    with torch.no_grad():
        logits_orig = model(x)[0, class_idx].item()
        logits_occ  = model(x_occluded)[0, class_idx].item()

    return float(logits_orig - logits_occ)  

# WYBÓR KLAS DO ANALIZY
TARGET_CLASSES = ["Vascular Lesions"] 
MAX_TRIES_PER_CLASS = 500      # bezpieczeństwo na wypadek dużych folderów

def pick_correct_sample_for_class(
    cls_name: str,
    val_root: Path,
    model: nn.Module,
    transform,
    class_to_idx: dict,
    device,
    max_tries: int = 500,
):
    cls_dir = val_root / cls_name
    if not cls_dir.exists():
        raise FileNotFoundError(f"Nie ma folderu klasy: {cls_dir}")

    # zbierz pliki obrazów
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    files = [p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if len(files) == 0:
        raise FileNotFoundError(f"Brak obrazów w: {cls_dir}")

    # opcjonalnie losowo, żeby nie brać zawsze tego samego
    random.shuffle(files)

    tries = 0
    for p in files:
        tries += 1
        if tries > max_tries:
            break

        try:
            img = Image.open(p).convert("RGB")
            inp = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.argmax(model(inp), dim=1).item()

            if pred == class_to_idx[cls_name]:
                return inp.detach().cpu(), str(p)

        except Exception as e:
            print(f"[WARN] Nie udało się przetworzyć {p}: {e}")

    raise RuntimeError(
        f"Nie znaleziono poprawnie sklasyfikowanego obrazu dla klasy '{cls_name}'. ")

selected_classes = []
correct_samples = {}   # {cls: tensor_cpu}
selected_paths = {}    # {cls: path}

for cls in TARGET_CLASSES:
    inp_cpu, img_path = pick_correct_sample_for_class(
        cls_name=cls,
        val_root=VAL_ROOT,
        model=model,
        transform=transform,
        class_to_idx=class_to_idx,
        device=device,
        max_tries=MAX_TRIES_PER_CLASS,
    )
    selected_classes.append(cls)
    correct_samples[cls] = inp_cpu
    selected_paths[cls] = img_path

print("Wybrane próbki (poprawnie sklasyfikowane):")
for cls in selected_classes:
    print(f" - {cls}: {selected_paths[cls]}")

# Przygotuj instancje metod
methods = []
if LRP_:
    print("Inicjalizacja LRP...")
    lrp_model = LRPResNet34(model_lrp, device)
    methods.append("LRP")
    
# Multi-layer CAM – trzy głębokie poziomy ResNet34
target_layers = {
    "layer1": model.layer1[-1].conv2,
    "layer2": model.layer2[-1].conv2,
    "layer3": model.layer3[-1].conv2,
}

gc = GradCAM(model, target_layers["layer1"])       # klasyczny Grad-CAM
sc = ScoreCAM(model, target_layers["layer1"])      # Score-CAM
lc = WeightedMultiLayerCAM(model, target_layers)           # multiscale Layer-CAM

methods.extend(["Grad-CAM", "Score-CAM", "Layer-CAM"])


if DeepSHAP_:
    print("Inicjalizacja DeepSHAP...")
    deep_shap = DeepLiftShap(model)
    layer_deepshap = LayerDeepLiftShap(model, model.layer1)
    methods.append("DeepSHAP")

# BUFOR NA ZNORMALIZOWANE MAPY I CZASY 
normalized_maps = {m: [] for m in methods}  
method_times    = {m: [] for m in methods}  # czasy generowania map (sekundy)
normalized_energy_maps = {m: [] for m in methods}


n_rows = len(selected_classes)
n_cols = 1 + len(methods)  

n_rows_vis = max(1, n_rows * 3)
fig, axes = plt.subplots(n_rows_vis, n_cols, figsize=(4*n_cols, 3*n_rows_vis))
if n_rows_vis == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows_vis == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

def normalize_energy(M, target_mean=0.15):
    M = np.array(M, dtype=np.float32)
    M = np.clip(M, 0, 1)

    # 1. Dopasowanie energii
    m1 = M.mean()
    if m1 < 1e-12:
        return np.zeros_like(M, dtype=np.float32)
    M = M * (target_mean / m1)

    # 2. Min-max
    mn, mx = M.min(), M.max()
    if mx - mn < 1e-12:
        M = np.zeros_like(M, dtype=np.float32)
    else:
        M = (M - mn) / (mx - mn)

    # 3. Drugie dopasowanie mean
    m2 = M.mean()
    if m2 < 1e-12:
        return np.zeros_like(M, dtype=np.float32)
    M = M * (target_mean / m2)

    # 4. Drugie min-max
    mn2, mx2 = M.min(), M.max()
    if mx2 - mn2 < 1e-12:
        M = np.zeros_like(M, dtype=np.float32)
    else:
        M = (M - mn2) / (mx2 - mn2)

    # 5. Trzecie dopasowanie mean
    m3 = M.mean()
    if m3 < 1e-12:
        return np.zeros_like(M, dtype=np.float32)
    M = M * (target_mean / m3)

    # 6. Clip na końcu
    M = np.clip(M, 0, 1)

    return M


class XAIAnalyzer:

    def __init__(self, orig_img_uint8, maps_01, maps_energy):
        self.orig = orig_img_uint8
        self.maps01 = maps_01
        self.mapsE  = maps_energy
        self.methods = list(maps_01.keys())
        self.H, self.W = orig_img_uint8.shape[:2]
    def laplacian_energy(self, hm):
        lap = cv2.Laplacian(hm, cv2.CV_32F)
        return float(np.mean(np.abs(lap)))

    def peak_ratio(self, hm):
        return float(hm.max() / (hm.mean() + 1e-8))

    def top_percent_energy(self, hm, percent):
        v = hm.flatten()
        total = v.sum() + 1e-8
        k = max(1, int(len(v) * (percent / 100)))
        idx = np.argpartition(-v, k-1)[:k]
        return float(v[idx].sum() / total)

    def coverage(self, hm, thr=0.5):
        return float((hm > thr).mean())

    def object_overlap(self, hm):
        gray = cv2.cvtColor(self.orig, cv2.COLOR_BGR2GRAY)
        leaf_mask = (gray < 230).astype(np.float32)
        return float((hm * leaf_mask).sum() / (hm.sum() + 1e-8))


    def pearson_corr(self, a, b):
        x = a.flatten().astype(np.float32)
        y = b.flatten().astype(np.float32)
        xm, ym = x.mean(), y.mean()
        num = np.sum((x - xm) * (y - ym))
        den = np.sqrt(np.sum((x - xm)**2) * np.sum((y - ym)**2)) + 1e-8
        return float(num / den)

    def cosine(self, a, b):
        x = a.flatten()
        y = b.flatten()
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))

    def ssim_map(self, a, b):
        return float(ssim(np.clip(a,0,1), np.clip(b,0,1), data_range=1.0))
    def occlusion_drop(self, input_tensor, hm, model, class_idx, device, k_frac=0.1):
        hm_abs = np.abs(hm)
        flat = hm_abs.flatten()
        n_pix = flat.size
        k = max(1, int(n_pix * k_frac))

        idx = np.argpartition(-flat, k-1)[:k]

        mask = np.ones((1,1,224,224), dtype=np.float32)
        mask.reshape(-1)[idx] = 0.0
        mask_t = torch.from_numpy(mask).to(device)

        x = input_tensor.clone().to(device)
        x_occ = x * mask_t

        with torch.no_grad():
            o0 = model(x)[0, class_idx].item()
            o1 = model(x_occ)[0, class_idx].item()

        return float(o0 - o1)

    def random_drop(self, input_tensor, model, class_idx, device, k_frac=0.1):
        mask = np.ones((1,1,224,224), dtype=np.float32)
        k = int(224*224 * k_frac)
        idx = np.random.choice(224*224, k, replace=False)
        mask.reshape(-1)[idx] = 0.0
        mask_t = torch.from_numpy(mask).to(device)

        x = input_tensor.clone().to(device)
        x_occ = x * mask_t

        with torch.no_grad():
            o0 = model(x)[0, class_idx].item()
            o1 = model(x_occ)[0, class_idx].item()

        return float(o0 - o1)

    def bounding_box(self, hm, thr=0.5):
        ys, xs = np.where(hm > thr)
        if len(xs) == 0:
            return None
        return xs.min(), ys.min(), xs.max(), ys.max()

    def draw_box(self, box, color=(0,255,0)):
        img = self.orig.copy()
        if box is None:
            return img
        x1,y1,x2,y2 = box
        cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
        return img

    def draw_contour(self, hm, thr=0.5):
        img = self.orig.copy()
        mask = hm > thr
        img[mask] = [255,0,0]
        return img

    def run_all(self, class_name):
        print("\n===== ANALIZA XAI =====")

        rows = []
        gc = self.maps01.get("Grad-CAM")

        for m in self.methods:
            hm01 = self.maps01[m]
            hmE  = self.mapsE[m]

            row = {
                "method": m,
                "entropy": compute_entropy(hmE),

                "top1":  self.top_percent_energy(hmE, 1),
                "top5":  self.top_percent_energy(hmE, 5),
                "top10": self.top_percent_energy(hmE, 10),
                "top20": self.top_percent_energy(hmE, 20),
                "pearson_gc": self.pearson_corr(hm01, gc),
                "cosine_gc":  self.cosine(hm01, gc),
                "ssim_gc":    self.ssim_map(hm01, gc),
            }

            rows.append(row)

            # WYPISUJEMY
            print(f"\n--- {m} ---")
            for k,v in row.items():
                if k!="method":
                    print(f"{k}: {v:.4f}")



        with open("xai_metrics.csv","w",newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # ZAPIS DO FOLDERU Analiza/metryk
        df = pd.DataFrame(rows)
        df.to_csv(METRYKI_DIR / "tabelaryczne_wyniki.csv", index=False)

        # macierze porównawcze ---
        pearson_matrix = pd.DataFrame(
            {m: [self.pearson_corr(self.maps01[m], self.maps01[n]) for n in self.methods] 
            for m in self.methods},
            index=self.methods
        )
        cosine_matrix = pd.DataFrame(
            {m: [self.cosine(self.maps01[m], self.maps01[n]) for n in self.methods] 
            for m in self.methods},
            index=self.methods
        )
        ssim_matrix = pd.DataFrame(
            {m: [self.ssim_map(self.maps01[m], self.maps01[n]) for n in self.methods] 
            for m in self.methods},
            index=self.methods
        )

        pearson_matrix.to_csv(METRYKI_DIR / "macierz_pearson.csv")
        cosine_matrix.to_csv(METRYKI_DIR / "macierz_cosine.csv")
        ssim_matrix.to_csv(METRYKI_DIR / "macierz_ssim.csv")

        plt.figure(figsize=(8,6))
        sns.heatmap(pearson_matrix, annot=True, cmap='viridis')
        plt.title("Podobieństwo metod (Pearson)")
        plt.savefig(METRYKI_DIR / "macierz_pearson.png", dpi=200, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,6))
        sns.heatmap(cosine_matrix, annot=True, cmap='viridis')
        plt.title("Podobieństwo metod (Cosine)")
        plt.savefig(METRYKI_DIR / "macierz_cosine.png", dpi=200, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,6))
        sns.heatmap(ssim_matrix, annot=True, cmap='viridis')
        plt.title("Podobieństwo metod (SSIM)")
        plt.savefig(METRYKI_DIR / "macierz_ssim.png", dpi=200, bbox_inches='tight')
        plt.close()

        # wykresy słupkowe dla każdej metryki 
        metrics = ["entropy","top1","top5","top10","top20"]
        for metric in metrics:
            plt.figure(figsize=(10,5))
            sns.barplot(x="method", y=metric, data=df)
            plt.title(f"Porównanie metod: {metric}")
            plt.xticks(rotation=45)
            plt.savefig(METRYKI_DIR / f"wykres_{metric}.png", dpi=200, bbox_inches='tight')
            plt.close()

        print("Zapisano pełny zestaw metryk do:", METRYKI_DIR)


        print("\nZapisano xai_metrics.csv")
        print("Zapisano *box.png i *contour.png dla każdej metody.")

METRYKI_DIR = OUT_DIR / "metryki"
METRYKI_DIR.mkdir(parents=True, exist_ok=True)



# GŁÓWNA PĘTLA WIZUALIZACJI 
for r, cls in enumerate(selected_classes):
    inp_cpu = correct_samples[cls]          # [1,3,224,224] CPU
    inp = inp_cpu.to(device)


    r_raw = 2 * r
    r_norm = r_raw + 1

    orig = tensor_to_uint8_rgb(inp)         # HWC uint8
    orig_uint8 = orig[..., ::-1]  # HWC BGR
    ax0 = axes[r_raw, 0]
    ax0.imshow(orig[..., ::-1])             # tylko podgląd RGB
    ax0.set_title(f"Oryginał: {cls}")
    ax0.axis('off')
    ax0b = axes[r_norm, 0]
    ax0b.imshow(orig[..., ::-1])
    ax0b.set_title(f"Oryginał (kopią): {cls}")
    ax0b.axis('off')

    c = 1  # aktualna kolumna (metody)

    # LRP
    if LRP_:
        print(f"Generowanie LRP dla klasy: {cls}")
        t0 = time.perf_counter()
        rel_maps = lrp_model.lrp(
            inp,
            class_to_idx[cls],
            rule="alphabeta",
            alpha=1.0,
            beta=0.0,
            return_all_layers=True
        )
        t1 = time.perf_counter()
        method_times["LRP"].append(t1 - t0)

        R = rel_maps['layer1'][0].detach().cpu().numpy().sum(0)

        R_raw = cv2.resize(R.astype(np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

        R_pos = np.clip(R, 0.0, None)
   
        vmax_pos = np.percentile(R_pos, 99) if np.any(R_pos > 0) else 0.0
        if vmax_pos > 0:
            R_pos_clip = np.clip(R_pos, 0.0, vmax_pos) / (vmax_pos + 1e-8)
        else:
            R_pos_clip = R_pos

        R_pos_smooth = cv2.GaussianBlur((R_pos_clip).astype(np.float32), (5, 5), 0)

        R_unit = cv2.resize(R_pos_smooth, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        normalized_maps["LRP"].append(R_unit.copy())


        vmax = max(np.abs(R_raw).max(), 1e-8)
        im = axes[r_raw, c].imshow(R_raw, cmap='bwr', vmin=-vmax, vmax=+vmax)
        axes[r_raw, c].set_title("LRP (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)


        im2 = axes[r_norm, c].imshow(R_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("LRP (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

    if GradCAM_:
        print(f"Generowanie CAM dla klasy: {cls}")

        # Grad-CAM
        t0 = time.perf_counter()
        h_grad = gc.generate(inp, class_to_idx[cls])  # 2D
        t1 = time.perf_counter()
        method_times["Grad-CAM"].append(t1 - t0)

        h_grad_raw = cv2.resize(np.array(h_grad, dtype=np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        h_grad_unit = normalize_to_01(h_grad, signed=False, out_size=(orig.shape[1], orig.shape[0]))
        normalized_maps["Grad-CAM"].append(h_grad_unit.copy())

        im = axes[r_raw, c].imshow(h_grad_raw, cmap='jet')
        axes[r_raw, c].set_title("Grad-CAM (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(h_grad_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("Grad-CAM (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

        # Score-CAM
        print(f"Generowanie Score-CAM dla klasy: {cls}")
        t0 = time.perf_counter()
        h_score = sc.generate(inp, class_to_idx[cls])
        t1 = time.perf_counter()
        method_times["Score-CAM"].append(t1 - t0)

        h_score_raw = cv2.resize(np.array(h_score, dtype=np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        h_score_unit = normalize_to_01(h_score, signed=False, out_size=(orig.shape[1], orig.shape[0]))
        normalized_maps["Score-CAM"].append(h_score_unit.copy())

        im = axes[r_raw, c].imshow(h_score_raw, cmap='jet')
        axes[r_raw, c].set_title("Score-CAM (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(h_score_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("Score-CAM (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

        # Layer-CAM
        print(f"Generowanie Layer-CAM dla klasy: {cls}")
        t0 = time.perf_counter()
        h_layer = lc.generate(inp, class_to_idx[cls])
        t1 = time.perf_counter()
        method_times["Layer-CAM"].append(t1 - t0)

        h_layer_raw = cv2.resize(np.array(h_layer, dtype=np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        h_layer_unit = normalize_to_01(h_layer, signed=False, out_size=(orig.shape[1], orig.shape[0]))
        normalized_maps["Layer-CAM"].append(h_layer_unit.copy())

        im = axes[r_raw, c].imshow(h_layer_raw, cmap='jet')
        axes[r_raw, c].set_title("Layer-CAM (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(h_layer_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("Layer-CAM (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1

    if DeepSHAP_:
        print(f"Generowanie DeepSHAP dla klasy: {cls}")
        baselines = get_real_image_baselines(VAL_ROOT, transform, num_baselines=max_baselines).to(device)

        t0 = time.perf_counter()
        explainer = locals().get('layer_deepshap', None)
        if explainer is None:
            explainer = LayerDeepLiftShap(model, model.layer1)

        attribution = explainer.attribute(inp, baselines=baselines, target=class_to_idx[cls])
        t1 = time.perf_counter()
        method_times["DeepSHAP"].append(t1 - t0)

        # uśredniamy po kanałach
        attr = attribution.squeeze().cpu().detach().mean(0).numpy()
        attr_raw = cv2.resize(attr.astype(np.float32), (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)

        attr_pos = np.clip(attr, 0.0, None)
        vmax_attr = np.percentile(attr_pos, 99) if np.any(attr_pos > 0) else 0.0
        if vmax_attr > 0:
            attr_clip = np.clip(attr_pos, 0.0, vmax_attr) / (vmax_attr + 1e-8)
        else:
            attr_clip = attr_pos
        attr_smooth = cv2.GaussianBlur(attr_clip.astype(np.float32), (5, 5), 0)
        attr_unit = cv2.resize(attr_smooth, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        normalized_maps["DeepSHAP"].append(attr_unit.copy())

        vmax = max(np.abs(attr_raw).max(), 1e-8)
        im = axes[r_raw, c].imshow(attr_raw, cmap='bwr', vmin=-vmax, vmax=+vmax)
        axes[r_raw, c].set_title("DeepSHAP (raw)")
        axes[r_raw, c].axis('off')
        fig.colorbar(im, ax=axes[r_raw, c], fraction=0.046, pad=0.04)

        im2 = axes[r_norm, c].imshow(attr_unit, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_norm, c].set_title("DeepSHAP (0-1)")
        axes[r_norm, c].axis('off')
        fig.colorbar(im2, ax=axes[r_norm, c], fraction=0.046, pad=0.04)
        c += 1
        # 3. WIERSZ: MAPY UNORMOWANE ENERGETYCZNIE
    r_energy = r_raw + 2  # trzeci wiersz dla tej próbki

    # oryginał w kolumnie 0
    ax0c = axes[r_energy, 0]
    ax0c.imshow(orig[..., ::-1])
    ax0c.set_title(f"Oryginał (energia): {cls}")
    ax0c.axis('off')


    c2 = 1
    for method_name in methods:
        hm = normalized_maps[method_name][-1]  
        

        # adaptacyjna normalizacja energii
        hm_energy = normalize_energy(hm, target_mean=0.15)
        normalized_energy_maps[method_name].append(hm_energy.copy())
        mean_energy = hm_energy.mean()
        print(f"[ENERGIA] {method_name}: mean = {mean_energy:.4f}")

        # rysowanie
        im3 = axes[r_energy, c2].imshow(hm_energy, cmap='viridis', vmin=0.0, vmax=1.0)
        axes[r_energy, c2].set_title(f"{method_name} (energia≈0.15)")
        axes[r_energy, c2].axis('off')
        fig.colorbar(im3, ax=axes[r_energy, c2], fraction=0.046, pad=0.04)

        c2 += 1


maps01 = {m: normalized_maps[m][0] for m in methods}
mapsE  = {m: normalized_energy_maps[m][0] for m in methods}

analyzer = XAIAnalyzer(orig_uint8, maps01, mapsE)
analyzer.run_all("Quercus")


plt.tight_layout()
save_path = METRYKI_DIR / f"wizualizacje_XAI_{len(selected_classes)}obrazy.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print(f"Zapisano: {save_path}")


