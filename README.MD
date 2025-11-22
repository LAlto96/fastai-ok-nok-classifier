# OK/NOK Image Classifier (fastai)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![fastai](https://img.shields.io/badge/fastai-vision-success)](https://docs.fast.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ce projet implémente un **classificateur binaire d’images (OK / NOK)** basé sur **fastai** et **PyTorch**, conçu pour des scénarios de **contrôle qualité**, de **filtrage automatisé d’images**, ou de **prétraitement visuel sécurisé**.

Il inclut l’ensemble de la pipeline machine learning :

- préparation et nettoyage des datasets  
- constitution d’un split train / validation / test  
- transfert d’apprentissage avec ResNet18  
- analyse complète des performances  
- optimisation du seuil de décision (threshold tuning)  
- outils d’inférence (prédiction sur image individuelle)  
- export du modèle pour un usage en production

---

Voici une version corrigée et cohérente de la **Structure du projet** pour ton `README.md`, en tenant compte de ce qui apparaît réellement dans ta repo (screenshots + arborescence GitHub).

Tu peux remplacer directement ton bloc actuel par celui-ci :

## 📁 Structure du projet

```
.
├── clothing-dataset-small-master/ # Dataset externe complet (non utilisé dans le pipeline final)
│ ├── test/
│ ├── train/
│ └── validation/
│
├── dataset/ # Dataset final (généré par le script)
│ ├── test/
│ ├── train/
│ └── val/
│
├── deepfashion/ # Données brutes extraites depuis DeepFashion
│ ├── annos/ # Annotations JSON/texte
│ └── image/ # Images extraites
│
├── nok/ # Images NOK collectées manuellement
├── ok/ # Images OK collectées manuellement
│
├── .gitattributes
├── .gitignore
├── LICENSE # Licence MIT (Ludovic Lafon)
├── nbdime_config.json # Configuration nbdime pour diff Jupyter
│
├── notebook.ipynb # Notebook principal d’entraînement fastai
├── test_model.ipynb # Notebook de test (inférence & batch prediction)
│
├── ok_nok_resnet18_fastai.pkl # Modèle exporté (fastai)
└── ok_nok_threshold.json # Seuil optimal calculé
```

---

## 📊 Objectif du modèle

Le modèle classe les images en deux catégories :

- **OK** : image valide (ex. vêtement isolé)  
- **NOK** : image invalide (ex. présence d’une personne)

Dans un contexte de contrôle qualité, les erreurs **NOK → OK** (faux positifs) sont critiques.  
Le projet inclut donc une étape d’optimisation du seuil de décision pour rendre le modèle plus **conservateur**.

---

## ✔️ Entraînement du modèle

L’entraînement repose sur :

- `DataBlock` fastai  
- architecture **ResNet18** pré-entraînée (ImageNet)  
- `fine_tune(5)`  
- augmentations légères  
- dataset équilibré  
- métriques : accuracy, precision, recall, F1

### **Performances sur le jeu de test**

- **Accuracy : ~0.99**  
- **Precision (OK) : 0.996**  
- **Recall (OK) : 0.977**  
- **F1-score (OK) : 0.986**

---

## 🎚️ Optimisation du seuil (Threshold Tuning)

En évaluant des thresholds entre **0.50** et **0.99**, le modèle montre que :

> **Threshold optimal = 0.99**

Ce seuil minimise fortement les **faux positifs (NOK → OK)** tout en conservant une excellente performance globale.

Le threshold est sauvegardé dans :  
`ok_nok_threshold.json`

---

## 🔍 Inférence sur une image

Une fonction d’inférence personnalisée applique le threshold afin d’obtenir une prédiction plus robuste :

```python
def predict_image(img_path, threshold=THRESHOLD_OK):
    img = PILImage.create(img_path)
    pred_label, pred_idx, probs = learn.predict(img)

    idx_ok = learn.dls.vocab.o2i["ok"]
    prob_ok = float(probs[idx_ok])

    final_label = "ok" if prob_ok >= threshold else "nok"

    display(img.to_thumb(256, 256))
    print(f"Probabilité 'ok' = {prob_ok:.4f}")
    print(f"Seuil utilisé     = {threshold:.2f}")
    print(f"Prédiction brute  = {pred_label}")
    print(f"Prédiction finale = {final_label}")

    return final_label, prob_ok
````

Usage :

```python
predict_image("path/to/image.jpg")
```

---

## 💾 Export du modèle

```python
learn.export("ok_nok_resnet18_fastai.pkl")
```

Sauvegarde du threshold :

```python
json.dump({"threshold_ok": 0.99}, open("ok_nok_threshold.json", "w"))
```

---

## 🔧 Installation

```bash
conda create -n oknok python=3.10
conda activate oknok
pip install fastai torch torchvision
```

Datasets à placer dans :

```
dataset/
    train/
    val/
    test/
```

---

## 🧪 Améliorations possibles

* essayer ResNet34 / ResNet50
* ajouter des “hard negatives” pour renforcer la robustesse
* augmenter les données (augmentation plus forte)
* packager une API (FastAPI)
* script CLI d’inférence

---

## 📜 Licence

Ce projet est distribué sous licence **MIT**.
Voir le fichier [`LICENSE`](LICENSE).