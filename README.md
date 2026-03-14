# XAI-Leaf-Skin: Interpretowalna Klasyfikacja Obrazów


**XAI-Leaf-Skin** to system oparty na sieciach głębokich (**ResNet34**), który zajmuje się klasyfikacją obrazów w dwóch domenach: dendrologicznej oraz medycznej. Projekt kładzie szczególny nacisk na **wyjaśnialność modelu**, wykorzystując zaawansowane metody XAI do wizualizacji cech obrazu determinujących decyzje modelu.

---

## Project Overview
Projekt realizuje zadanie automatycznego rozpoznawania obiektów na zdjęciach, wspierając użytkownika interpretacją wizualną w postaci map ciepła (heatmaps). System analizuje:

* **Bazę liści drzew**: Klasyfikacja 14 gatunków (m.in. *Quercus, Acer, Alnus*) na podstawie morfologii blaszki liściowej.
![leaf database](img/lisc.jpg) 
* **Bazę zmian skórnych**: Diagnostyka różnicowa m.in. czerniaka (*Melanoma*) oraz zmian naczyniowych (*Vascular Lesions*).
![skin database](img/skora.jpg)

### Key Technologies:
* **Architektura**: ResNet34.
* **Interpretowalność**: Grad-CAM, Score-CAM, Layer-CAM, LRP, DeepSHAP.
* **Metryki XAI**: Entropia, Koncentracja Energii, SSIM, Korelacja Pearsona.
* **Środowisko**: PyTorch, Captum, OpenCV, Scikit-learn.

---

## 🛠️ Analyzed Methods (XAI)
W projekcie zaimplementowano i porównano zaawansowane metody wizualizacji atrybucji:



---

## 📂 Project Structure
* **`1.py` – Trening i Diagnostyka**: Inicjalizacja modelu oraz generowanie metryk (t-SNE, Confusion Matrix).
* **`2.py` – Analiza Warstwowa**: Narzędzie do wizualnego porównania metod CAM na różnych etapach architektury sieci.
* **`3.py` – Ewaluacja Metryk XAI**: Zaawansowany moduł obliczający twarde metryki jakości wyjaśnień oraz generujący zbiorcze porównania metod.

---

## 🚀 Getting Started
Projekt wymaga środowiska Python 3.8+ oraz bibliotek wymienionych w skryptach.
