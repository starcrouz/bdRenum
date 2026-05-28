import os
import shutil
import csv
import io
import math
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Utilisation du client Google GenAI
from google import genai
from google.genai import types

# --- Configuration ---
SOURCE_DIR = r"F:\Téléchargements\_BD\titi"
DEST_DIR = r"F:\Téléchargements\_BD\titi-renommes"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

VALID_EXTENSIONS = ('.cbz', '.cbr', '.pdf', '.epub')
BATCH_SIZE = 50

# --- Prompt système pour l'extraction des métadonnées ---
SYSTEM_INSTRUCTION = """Vous êtes un assistant spécialisé dans le traitement et le renommage de fichiers de bandes dessinées (BD).
Votre tâche consiste à recevoir une liste de noms de fichiers (un par ligne) et à extraire les métadonnées structurées sous forme de tableau CSV.

Format de sortie :
- Ne retournez QUE le bloc CSV entouré de ```csv et ```. Aucun autre texte d'introduction, de conclusion ou d'explication.
- Le séparateur doit être le point-virgule (;) et chaque champ doit être entouré de guillemets doubles (").
- L'en-tête doit être exactement : "Fichier";"Série";"Numéro";"Titre";"Année";"Divers"

Règles de parsing pour chaque fichier :
1. "Fichier" : Le nom exact du fichier d'origine tel que fourni en entrée (avec son extension).
2. "Série" : Le nom propre de la série de bande dessinée.
   - Supprimez les underscores (_) et remplacez-les par des espaces.
   - Si l'article principal est rejeté à la fin (ex: "Dinodyssée", "Arabe_Du_Futur_L'"), reformatez-le pour le remettre au début (ex: "L'arabe du futur", "La dinodyssée").
   - Uniformisez la casse (ex: "Amère Russie" au lieu de "AMERE_RUSSIE").
3. "Numéro" : Le numéro du tome de la BD (ex: "01", "02", "12").
   - Si c'est une intégrale, écrivez "INT".
   - Si c'est un "One Shot" (tome unique), laissez ce champ vide.
4. "Titre" : Le titre spécifique de cet album/tome. Si le titre n'est pas mentionné ou s'il s'agit simplement du nom de la série, laissez vide.
5. "Année" : L'année de publication sur 4 chiffres (ex: "2024"). Laissez vide si absente.
6. "Divers" : Contient les métadonnées résiduelles séparées par des virgules :
   - Auteurs/dessinateurs présents (généralement entre parenthèses, ex: "(Zabus et al.)" ou "Gaet's").
   - Le tag de release/communauté si présent (ex: "@N_art_BD_FR", "@BD_fr").
   - Mention "OS" ou "One Shot" si elle est mentionnée dans le nom du fichier.
"""

def list_bd_files(directory):
    filenames = []
    try:
        for item in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, item)) and item.lower().endswith(VALID_EXTENSIONS):
                filenames.append(item)
    except FileNotFoundError:
        print(f"ERREUR: Le dossier source '{directory}' n'a pas été trouvé.")
        return None
    return filenames

def call_gemini_api(filenames, batch_num=1, total_batches=1):
    if not filenames:
        print("Aucun fichier à envoyer à l'API Gemini.")
        return None

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("ERREUR: La clé API Gemini n'est pas configurée dans le fichier .env.")
        return None

    payload_content = "\n".join(filenames)
    
    print(f"\n=== Lot {batch_num}/{total_batches} : {len(filenames)} fichiers ===")
    
    # TRACE : Afficher la liste des fichiers du lot
    print(f"📋 Fichiers envoyés à l'API Gemini :")
    for i, filename in enumerate(filenames, 1):
        print(f"  {i:2d}. {filename}")
    print("-" * 80)
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"🚀 Envoi du lot {batch_num} à l'API Gemini (gemini-2.5-flash)...")
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=payload_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
            )
        )
        
        csv_content = response.text
        if not csv_content:
            print("ERREUR: Réponse vide de l'API Gemini.")
            return None

        # Nettoyage du CSV
        if isinstance(csv_content, str):
            if csv_content.startswith("```csv\n"):
                csv_content = csv_content[len("```csv\n"):]
            if csv_content.endswith("\n```"):
                csv_content = csv_content[:-len("\n```")]
            if csv_content.startswith("```\n"): 
                csv_content = csv_content[len("```\n"):]
        else:
            print(f"ERREUR: Le contenu CSV attendu n'est pas une chaîne de caractères. Reçu: {type(csv_content)}")
            return None
        
        print(f"✅ Réponse CSV reçue pour le lot {batch_num}")
        return csv_content.strip()

    except Exception as e:
        print(f"❌ ERREUR lors de l'appel à l'API Gemini pour le lot {batch_num}: {e}")
        return None

def parse_csv_response(csv_content, batch_num=1):
    """Parse la réponse CSV et retourne une liste de dictionnaires."""
    if not csv_content:
        return []
    
    parsed_data = []
    csvfile = io.StringIO(csv_content)
    
    try:
        csvfile.seek(0)
        
        try:
            sample = csvfile.read(2048)
            csvfile.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(csvfile, dialect=dialect)
        except csv.Error:
            csvfile.seek(0)
            reader = csv.DictReader(csvfile, delimiter=';')

        for row in reader:
            cleaned_row = {}
            for k, v in row.items():
                if k is not None:
                    clean_key = str(k).strip().strip('"') if k else ""
                    clean_value = str(v).strip().strip('"') if v is not None else ""
                    cleaned_row[clean_key] = clean_value
            
            if cleaned_row and any(v for v in cleaned_row.values()):
                parsed_data.append(cleaned_row)
                
    except Exception as e:
        print(f"ERREUR lors du parsing du CSV du lot {batch_num}: {e}")
        return []
    
    print(f"Lot {batch_num}: {len(parsed_data)} entrées parsées.")
    return parsed_data

def sanitize_filename_part(part):
    """Supprime les caractères invalides pour les noms de fichiers Windows et nettoie."""
    if not part:
        return ""
    invalid_chars = r'<>:"/\|?*' + "".join(chr(i) for i in range(32))
    sanitized = "".join(c for c in part if c not in invalid_chars)
    sanitized = sanitized.strip(". ") 
    return sanitized.strip() 

def normalize_filename(filename):
    """Normalise un nom de fichier en réduisant les espaces multiples."""
    import re
    return re.sub(r'\s+', ' ', filename)

def process_files(all_parsed_data, original_files_in_source):
    """Copie et renomme les fichiers BD dans le dossier de résultat."""
    if not all_parsed_data:
        print("Aucune donnée parsée à traiter.")
        return

    # Créer le dossier de destination
    if not os.path.exists(DEST_DIR):
        try:
            os.makedirs(DEST_DIR)
            print(f"📁 Dossier de destination '{DEST_DIR}' créé.")
        except OSError as e:
            print(f"ERREUR: Impossible de créer le dossier de destination '{DEST_DIR}': {e}")
            return

    processed_count = 0
    error_count = 0
    
    # Créer un mapping global avec tous les lots
    metadata_map = {}
    for data_item in all_parsed_data:
        if 'Fichier' in data_item:
            metadata_map[data_item['Fichier']] = data_item
    
    # Créer une correspondance approximative pour les noms de fichiers avec espaces multiples
    normalized_map = {}
    for csv_filename, data in metadata_map.items():
        normalized_csv = normalize_filename(csv_filename)
        normalized_map[normalized_csv] = (csv_filename, data)
    
    # Trouver les correspondances
    found_matches = {}
    missing_in_csv = []
    
    for source_file in original_files_in_source:
        if source_file in metadata_map:
            found_matches[source_file] = metadata_map[source_file]
        else:
            normalized_source = normalize_filename(source_file)
            if normalized_source in normalized_map:
                csv_filename, data = normalized_map[normalized_source]
                found_matches[source_file] = data
                print(f"🔗 Correspondance approximative: '{source_file}' -> '{csv_filename}'")
            else:
                missing_in_csv.append(source_file)
    
    if missing_in_csv:
        print(f"\n⚠️  Fichiers non traités ({len(missing_in_csv)}):")
        for filename in missing_in_csv:
            print(f"   - {filename}")

    print(f"\n📝 Traitement de {len(found_matches)} fichiers...")

    for original_filename_from_fs in original_files_in_source:
        if original_filename_from_fs not in found_matches:
            print(f"❌ IGNORÉ: '{original_filename_from_fs}'")
            error_count += 1
            continue
        
        data = found_matches[original_filename_from_fs]
        original_full_path = os.path.join(SOURCE_DIR, original_filename_from_fs)
        
        try:
            _ , extension = os.path.splitext(original_filename_from_fs)
            
            serie_raw = data.get("Série", "")
            tome_raw = data.get("Numéro", data.get("Tome", ""))
            album_raw = data.get("Titre", data.get("Album", ""))
            annee_raw = data.get("Année", "")
            divers_raw = data.get("Divers", "")

            s_clean = sanitize_filename_part(serie_raw)
            if not s_clean:
                print(f"❌ ERREUR: Série vide pour '{original_filename_from_fs}'")
                error_count += 1
                continue
            
            t_clean = sanitize_filename_part(tome_raw)
            a_csv_clean = sanitize_filename_part(album_raw) 
            y_clean = sanitize_filename_part(annee_raw)
            d_csv_clean = sanitize_filename_part(divers_raw)

            # Vérifier si c'est un OS (One Shot)
            is_one_shot = False
            
            if d_csv_clean and ('OS' in d_csv_clean.upper() or 'ONE SHOT' in d_csv_clean.upper()):
                is_one_shot = True
            elif (a_csv_clean and s_clean and 
                  a_csv_clean.lower().strip() == s_clean.lower().strip() and 
                  not t_clean):
                is_one_shot = True

            # Construction du nom
            name_components = [s_clean] 

            if is_one_shot:
                divers_section_parts = []
                
                if (a_csv_clean and s_clean and 
                    a_csv_clean.lower().strip() != s_clean.lower().strip()):
                    divers_section_parts.append(a_csv_clean)
                
                if d_csv_clean:
                    d_final_for_list = d_csv_clean.strip(", ")
                    if d_final_for_list:
                        divers_section_parts.append(d_final_for_list)
                
                main_part = s_clean
                
            else:
                if t_clean:
                    name_components.append(f"- {t_clean}")
                
                if a_csv_clean: 
                    name_components.append(f"- {a_csv_clean}")
                
                main_part = " ".join(name_components)
                
                divers_section_parts = []
                if d_csv_clean: 
                    d_final_for_list = d_csv_clean.strip(", ")
                    if d_final_for_list: 
                        divers_section_parts.append(d_final_for_list)
            
            if y_clean:
                main_part += f" ({y_clean})"
            
            if divers_section_parts:
                valid_divers_elements = [part for part in divers_section_parts if part]
                if valid_divers_elements:
                    main_part += f" [{', '.join(valid_divers_elements)}]"

            new_filename_base = " ".join(main_part.split())

            if not new_filename_base or new_filename_base == extension.lstrip('.'):
                print(f"❌ ERREUR: Nom invalide pour '{original_filename_from_fs}'")
                error_count += 1
                continue

            new_filename = f"{new_filename_base}{extension}"
            dest_full_path = os.path.join(DEST_DIR, new_filename)

            # COPIE ET RENOMMAGE RÉEL
            print(f"✅ Renommage: '{original_filename_from_fs}' -> '{new_filename}'")
            shutil.copy2(original_full_path, dest_full_path)
            processed_count += 1

        except Exception as e:
            print(f"❌ ERREUR lors du traitement de '{original_filename_from_fs}': {e}")
            error_count += 1
            
    print(f"\n🎯 === RÉSULTATS FINAUX ===")
    print(f"✅ {processed_count} fichiers copiés et renommés avec succès")
    print(f"❌ {error_count} erreurs ou fichiers ignorés")
    if processed_count > 0:
        print(f"📂 Fichiers disponibles dans: {DEST_DIR}")

# --- Exécution principale avec pagination ---
if __name__ == "__main__":
    print("--- Script de renommage de BD avec pagination ---")
    print(f"Configuration: {BATCH_SIZE} fichiers par lot\n")

    bd_filenames = list_bd_files(SOURCE_DIR)
    if not bd_filenames:
        print("Aucun fichier BD trouvé. Arrêt du script.")
        exit()

    total_files = len(bd_filenames)
    total_batches = math.ceil(total_files / BATCH_SIZE)
    
    print(f"📁 {total_files} fichiers trouvés dans '{SOURCE_DIR}'")
    print(f"📦 Traitement en {total_batches} lot(s) de {BATCH_SIZE} fichiers maximum")
    
    all_parsed_data = []
    
    # Traiter par lots
    for batch_num in range(1, total_batches + 1):
        start_idx = (batch_num - 1) * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_files = bd_filenames[start_idx:end_idx]
        
        print(f"\n🔄 Traitement du lot {batch_num}/{total_batches}")
        
        csv_response = call_gemini_api(batch_files, batch_num, total_batches)
        
        if csv_response:
            parsed_data = parse_csv_response(csv_response, batch_num)
            all_parsed_data.extend(parsed_data)
        else:
            print(f"❌ Échec du traitement du lot {batch_num}")
    
    if all_parsed_data:
        print(f"\n📊 Total: {len(all_parsed_data)} entrées parsées sur {total_files} fichiers")
        process_files(all_parsed_data, bd_filenames)
    else:
        print("❌ Aucune donnée reçue de l'API Gemini.")

    print("\n--- Fin du script ---")