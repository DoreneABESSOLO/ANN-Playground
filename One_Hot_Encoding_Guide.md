# Guide du One-Hot Encoding pour votre Projet ANN

## Table des matières
1. [Qu'est-ce que le One-Hot Encoding ?](#quest-ce-que-le-one-hot-encoding)
2. [Quand utiliser le One-Hot Encoding ?](#quand-utiliser-le-one-hot-encoding)
3. [Application à votre projet](#application-à-votre-projet)
4. [Implémentation pratique](#implémentation-pratique)
5. [Exemples de code](#exemples-de-code)

---

## Qu'est-ce que le One-Hot Encoding ?

Le **One-Hot Encoding** (encodage one-hot) est une technique de préparation des données qui convertit des variables catégorielles en vecteurs binaires. Chaque catégorie devient une colonne binaire (0 ou 1), où seule la colonne correspondant à la catégorie de l'observation est à 1, les autres sont à 0.

### Exemple simple

**Avant (variable catégorielle) :**
```
Target
------
Graduate
Dropout
Enrolled
Graduate
```

**Après (one-hot encoding) :**
```
Target_Graduate  Target_Dropout  Target_Enrolled
----------------  --------------  ---------------
       1                0               0
       0                1               0
       0                0               1
       1                0               0
```

### Avantages
- ✅ Élimine l'ordre arbitraire entre catégories
- ✅ Compatible avec les algorithmes de machine learning
- ✅ Permet au modèle de traiter chaque catégorie indépendamment
- ✅ Nécessaire pour la cross-entropy avec softmax

### Inconvénients
- ❌ Augmente la dimensionnalité (problème si beaucoup de catégories)
- ❌ Peut créer de la multicollinéarité (résolu en supprimant une colonne)

---

## Quand utiliser le One-Hot Encoding ?

### ✅ **OBLIGATOIRE pour la TARGET (variable dépendante)**

Dans votre cas, la variable **Target** avec 3 classes (Graduate, Dropout, Enrolled) **DOIT** être encodée en one-hot car :
- Votre modèle utilise **softmax** en sortie
- La fonction de perte **cross-entropy** nécessite des labels one-hot
- Le calcul du gradient dans `backward()` utilise `y_pred - y_true` qui suppose un format one-hot

### ✅ **Recommandé pour les FEATURES catégorielles nominales**

Variables catégorielles **sans ordre** avec **peu de modalités** (< 10-15) :
- `Gender` (2 modalités : M, F)
- `Debtor` (2 modalités : Yes, No)
- `Scholarship holder` (2 modalités : Yes, No)
- `Tuition fees up to date` (2 modalités : Yes, No)

### ⚠️ **À considérer pour les FEATURES catégorielles avec beaucoup de modalités**

Variables avec **beaucoup de catégories** (> 15) :
- `Application mode` (plusieurs modalités)
- `Course` (plusieurs modalités)
- `Nacionality` (plusieurs modalités)

**Alternatives :**
- **Label Encoding** : si les codes numériques n'ont pas d'ordre
- **Embedding** : pour des catégories très nombreuses (avancé)

### ❌ **PAS pour les variables numériques**

Variables continues comme :
- `Age at enrollment`
- `Admission grade`
- `Curricular units 1st sem (grade)`

**Action :** Utiliser **normalisation/standardisation** (StandardScaler, MinMaxScaler)

---

## Application à votre projet

### Structure de vos données

D'après votre analyse exploratoire :

**Target (variable dépendante) :**
- 3 classes : `Graduate` (49.93%), `Dropout` (32.12%), `Enrolled` (17.95%)
- **Format actuel :** Catégorielle (object)
- **Format requis :** One-hot encoding → shape `[batch_size, 3]`

**Features catégorielles identifiées :**
- `Gender` (2 modalités)
- `Debtor` (2 modalités)
- `Scholarship holder` (2 modalités)
- `Tuition fees up to date` (2 modalités)
- `Application mode` (plusieurs modalités)
- `Course` (plusieurs modalités)
- Variables avec classes créées : `Mother_Occupation_Class`, `Father_Occupation_Class`, etc.

**Features numériques :**
- Variables continues (grades, ages, rates, etc.)
- **Action :** Normalisation, pas de one-hot encoding

---

## Implémentation pratique

### Workflow recommandé

```
1. Séparer Target et Features
   ↓
2. One-hot encoder la TARGET (obligatoire)
   ↓
3. Identifier les features catégorielles
   ↓
4. One-hot encoder les features catégorielles avec peu de modalités
   ↓
5. Label encoder ou garder tel quel les features avec beaucoup de modalités
   ↓
6. Normaliser les features numériques
   ↓
7. Combiner toutes les features
   ↓
8. Vérifier les dimensions pour l'entrée du modèle
```

### Dimensions attendues

- **Input (X) :** `[batch_size, nombre_de_features]`
- **Target (y) :** `[batch_size, 3]` (one-hot)

Le paramètre `initial_input_size` dans votre `ANN.__init__()` doit correspondre au nombre total de features après encodage.

---

## Exemples de code

### Méthode 1 : Pandas `get_dummies()` (Simple et rapide)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Charger vos données
# df = votre DataFrame

# ============================================
# 1. ONE-HOT ENCODING DE LA TARGET (OBLIGATOIRE)
# ============================================
y_onehot = pd.get_dummies(df['Target'], prefix='Target')
print(f"Shape de y_onehot: {y_onehot.shape}")  # Devrait être (4424, 3)
print(f"Colonnes: {y_onehot.columns.tolist()}")

# Convertir en numpy array pour votre modèle
y = y_onehot.values  # Shape: (4424, 3)

# ============================================
# 2. PRÉPARATION DES FEATURES
# ============================================

# Séparer les features de la target
X_df = df.drop(columns=['Target'])

# Identifier les colonnes catégorielles avec peu de modalités
categorical_cols_few = ['Gender', 'Debtor', 'Scholarship holder', 'Tuition fees up to date']
# Vérifier le nombre de modalités
for col in categorical_cols_few:
    if col in X_df.columns:
        n_unique = X_df[col].nunique()
        print(f"{col}: {n_unique} modalités")

# One-hot encoder les features catégorielles avec peu de modalités
X_encoded = pd.get_dummies(
    X_df, 
    columns=categorical_cols_few,
    prefix=categorical_cols_few,
    drop_first=False  # Garder toutes les colonnes (ou True pour éviter multicollinéarité)
)

# Pour les variables avec beaucoup de modalités, garder label encoding
# (si elles sont déjà numériques) ou appliquer label encoding
high_cardinality_cols = ['Application mode', 'Course', 'Nacionality']
for col in high_cardinality_cols:
    if col in X_encoded.columns and X_encoded[col].dtype == 'object':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# ============================================
# 3. NORMALISATION DES FEATURES NUMÉRIQUES
# ============================================

# Identifier les colonnes numériques restantes
numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns.tolist()

# Normaliser
scaler = StandardScaler()
X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

# ============================================
# 4. CONVERSION FINALE EN NUMPY
# ============================================

X = X_encoded.values  # Shape: (4424, nombre_de_features)
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")

# Vérifier que initial_input_size correspond
initial_input_size = X.shape[1]
print(f"initial_input_size pour votre modèle: {initial_input_size}")
```

### Méthode 2 : Scikit-learn `OneHotEncoder` (Plus robuste pour production)

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
import numpy as np

# ============================================
# 1. TARGET : One-Hot Encoding
# ============================================
target_encoder = OneHotEncoder(sparse_output=False, drop=None)
y_onehot = target_encoder.fit_transform(df[['Target']])
print(f"Shape de y: {y_onehot.shape}")  # (4424, 3)

# Obtenir les noms des classes
class_names = target_encoder.categories_[0]
print(f"Classes: {class_names}")

# ============================================
# 2. FEATURES : Préparation
# ============================================
X_df = df.drop(columns=['Target'])

# Séparer les types de colonnes
categorical_cols = ['Gender', 'Debtor', 'Scholarship holder', 'Tuition fees up to date']
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
high_cardinality_cols = [col for col in X_df.columns 
                         if col not in categorical_cols + numeric_cols 
                         and X_df[col].dtype == 'object']

# One-hot encoder les catégorielles avec peu de modalités
if categorical_cols:
    ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' évite multicollinéarité
    X_categorical = ohe.fit_transform(X_df[categorical_cols])
    categorical_feature_names = ohe.get_feature_names_out(categorical_cols)
    print(f"Features catégorielles encodées: {X_categorical.shape[1]} colonnes")
else:
    X_categorical = np.array([]).reshape(len(X_df), 0)
    categorical_feature_names = []

# Label encoder les catégorielles avec beaucoup de modalités
X_high_card = np.zeros((len(X_df), len(high_cardinality_cols)))
for i, col in enumerate(high_cardinality_cols):
    le = LabelEncoder()
    X_high_card[:, i] = le.fit_transform(X_df[col].astype(str))

# Normaliser les numériques
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X_df[numeric_cols])

# ============================================
# 3. COMBINER TOUTES LES FEATURES
# ============================================
X = np.hstack([X_categorical, X_high_card, X_numeric])
print(f"Shape finale de X: {X.shape}")
print(f"Nombre total de features: {X.shape[1]}")
```

### Méthode 3 : Pipeline complet avec votre modèle

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from activation_fonctions import Activation_Fonctions
from MLP import ANN

# Charger et préparer les données
# ... (code de préparation ci-dessus) ...

# Créer le modèle
activation = Activation_Fonctions()
model = ANN(
    learning_rate=0.01,
    size_of_layers=[64, 32],  # Exemple : 2 couches cachées
    initial_input_size=X.shape[1],  # Nombre de features après encodage
    activation_fonction=activation.ReLU_activation
)

# Entraîner
print("Début de l'entraînement...")
model.fit(X, y)  # y doit être en format one-hot

# Prédire
predictions = model.predict(X)
print(f"Shape des prédictions: {predictions.shape}")  # (4424, 3)

# Convertir les prédictions one-hot en classes
predicted_classes = np.argmax(predictions, axis=1)
class_names = ['Dropout', 'Enrolled', 'Graduate']  # Vérifier l'ordre
predicted_labels = [class_names[i] for i in predicted_classes]
```

---

## Points importants à retenir

### ✅ Pour la TARGET
- **One-hot encoding OBLIGATOIRE**
- Format : `[batch_size, 3]`
- Ordre des colonnes : Vérifier l'ordre (Dropout, Enrolled, Graduate)

### ✅ Pour les FEATURES
- **Catégorielles avec peu de modalités (< 10)** : One-hot encoding
- **Catégorielles avec beaucoup de modalités (> 15)** : Label encoding ou garder tel quel
- **Numériques** : Normalisation (StandardScaler), PAS de one-hot

### ⚠️ Attention
- Vérifier que `initial_input_size` correspond au nombre total de features après encodage
- Si vous utilisez `drop_first=True` dans OneHotEncoder, vous aurez une colonne de moins par variable
- Sauvegarder les encoders pour pouvoir transformer les nouvelles données de la même manière

### 📊 Vérification finale

Avant d'entraîner votre modèle, vérifiez :

```python
print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"initial_input_size dans ANN: {X.shape[1]}")
print(f"Nombre de classes dans y: {y.shape[1]}")
assert y.shape[1] == 3, "y doit avoir 3 colonnes (one-hot pour 3 classes)"
assert np.allclose(y.sum(axis=1), 1.0), "Chaque ligne de y doit sommer à 1"
```

---

## Ressources supplémentaires

- [Scikit-learn OneHotEncoder Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Pandas get_dummies Documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- [Cross-Entropy Loss avec Softmax](https://en.wikipedia.org/wiki/Cross_entropy)

---

**Note :** Ce guide est spécifiquement adapté à votre projet de prédiction du décrochage et de la réussite académique des étudiants. Adaptez les noms de colonnes selon votre dataset exact.

