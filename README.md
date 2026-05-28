# renommeur_bd

Un script Python automatisé pour renommer proprement des fichiers de bandes dessinées (BD) en utilisant l'intelligence artificielle de **Google Gemini**.

Le script extrait automatiquement la série, le numéro de tome, le titre de l'album, l'année de publication et les informations diverses à partir des noms de fichiers d'origine (souvent mal formatés), puis copie et renomme les fichiers vers un dossier de destination.

## Fonctionnalités

*   **Extraction intelligente** : Utilise Gemini pour extraire les métadonnées (Série, Tome, Titre, Année, Divers) à partir de noms de fichiers complexes.
*   **Traitement par lots (Pagination)** : Envoie les fichiers par lots de 50 pour respecter les limites et optimiser la rapidité.
*   **Détection automatique des "One Shot"** : Adapte la structure du nom final si la BD est un tome unique.
*   **Sécurisé & Propre** : La clé API est stockée de manière sécurisée dans un fichier `.env`. Les fichiers d'origine sont copiés et non écrasés.
*   **Compatibilité Windows** : Nettoie automatiquement les caractères interdits pour les noms de fichiers sous Windows (`< > : " / \ | ? *`).

## Installation et Configuration

### 1. Installation des dépendances

Assurez-vous d'avoir Python installé. Dans votre terminal, activez votre environnement virtuel et installez les dépendances nécessaires :

```powershell
# Activer l'environnement virtuel
.\venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### 2. Configuration via le fichier `.env`

Copiez le fichier d'exemple [.env.example](file:///c:/Users/steph/Documents/bdRenum/.env.example) pour créer votre propre fichier `.env` à la racine :

```powershell
cp .env.example .env
```

Ouvrez ensuite le fichier `.env` et ajustez ses valeurs :

```env
# Clé API Google Gemini
GEMINI_API_KEY=votre_cle_api_gemini_ici

# Modèle Gemini à utiliser (ex: gemini-2.5-flash)
GEMINI_MODEL=gemini-2.5-flash

# Chemins des dossiers (utilisez des chemins absolus)
SOURCE_DIR=C:\chemin\vers\votre\dossier\source
DEST_DIR=C:\chemin\vers\votre\dossier\destination
```

*(Note : Si vous ne spécifiez pas ces variables dans le fichier `.env`, le script utilisera par défaut les chemins et le modèle configurés comme valeurs de secours à l'intérieur du script.)*

## Utilisation

Pour lancer le renommage, exécutez simplement la commande suivante :

```powershell
python renommeur_bd.py
```

*Note : Si vous rencontrez des problèmes d'affichage d'émojis dans la console Windows, vous pouvez forcer l'encodage en UTF-8 :*
```powershell
$env:PYTHONIOENCODING="utf-8"; python renommeur_bd.py
```
