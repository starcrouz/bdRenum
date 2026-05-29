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
SOURCE_DIR = os.environ.get("SOURCE_DIR", r"F:\Téléchargements\_BD\titi")
DEST_DIR = os.environ.get("DEST_DIR", r"F:\Téléchargements\_BD\titi-renommes")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

VALID_EXTENSIONS = ('.cbz', '.cbr', '.pdf', '.epub')
BATCH_SIZE = 50

# --- Prompt système pour l'extraction des métadonnées ---
SYSTEM_INSTRUCTION = """Tu reçois une liste numérotée de fichiers de BD (sous la forme "ID. Nom_de_fichier").
Extrais leurs métadonnées au format CSV avec le séparateur ";" et les valeurs entre guillemets.
Ne renvoie QUE le bloc de code CSV (entouré de ```csv ... ```). N'explique rien.

Colonnes du CSV : "ID";"Série";"Numéro";"Titre";"Année";"Divers"

Règles d'extraction :
- ID : Le numéro correspondant au fichier reçu en entrée (ex: "1", "2").
- Série : Nom de la série propre (sans underscore, article remis au début, ex: "Le prisonnier sans frontières").
- Numéro : Numéro du tome (ex: "03"). Traite les intégrales (ex: "INT2") comme un numéro de tome. Laisse vide si One Shot (OS).
- Titre : Nom de l'album / du tome (laisser vide si inexistant).
- Année : Année de publication sur 4 chiffres (laisser vide si absente).
- Divers : Auteurs/dessinateurs, mention "OS". Exclure absolument les tags de release comme @BD_fr ou @N_art_BD_FR.
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

def call_gemini_api(filenames, batch_num=1, total_batches=1, start_id=1):
    if not filenames:
        print("Aucun fichier à envoyer à l'API Gemini.")
        return None

    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("ERREUR: La clé API Gemini n'est pas configurée dans le fichier .env.")
        return None

    # Numéroter les fichiers pour le prompt de l'API (ex: "1. Fichier.cbz")
    payload_lines = [f"{start_id + i}. {filename}" for i, filename in enumerate(filenames)]
    payload_content = "\n".join(payload_lines)
    
    print(f"\n=== Lot {batch_num}/{total_batches} : {len(filenames)} fichiers ===")
    
    # TRACE : Afficher la liste des fichiers du lot
    print(f"📋 Fichiers envoyés à l'API Gemini :")
    for i, filename in enumerate(filenames):
        print(f"  {start_id + i:2d}. {filename}")
    print("-" * 80)
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print(f"🚀 Envoi du lot {batch_num} à l'API Gemini...")
        
        response = client.models.generate_content(
            model=GEMINI_MODEL,
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
    
    # Associer les métadonnées aux fichiers via l'ID (index dans original_files_in_source)
    found_matches = {}
    missing_in_csv = []
    processed_indices = set()
    
    for data_item in all_parsed_data:
        id_str = data_item.get('ID', '')
        try:
            # 1-indexed dans le CSV -> 0-indexed dans la liste python
            file_idx = int(id_str) - 1
            if 0 <= file_idx < len(original_files_in_source):
                source_file = original_files_in_source[file_idx]
                found_matches[source_file] = data_item
                processed_indices.add(file_idx)
        except (ValueError, TypeError):
            print(f"⚠️ ID invalide ignoré dans le CSV : {id_str}")
            
    # Identifier les fichiers non traités
    for idx, source_file in enumerate(original_files_in_source):
        if idx not in processed_indices:
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

            # 1. Extraction et nettoyage de l'auteur depuis le champ "Divers"
            author_parts = []
            if d_csv_clean:
                parts = [p.strip() for p in d_csv_clean.split(",")]
                for p in parts:
                    p_upper = p.upper()
                    # Ignorer les tags de release comme @... ou les mentions d'OS / One Shot
                    if p.startswith("@") or "OS" in p_upper or "ONE SHOT" in p_upper:
                        continue
                    # Nettoyer les parenthèses éventuelles et les espaces
                    cleaned_p = p.strip("() ").strip()
                    if cleaned_p:
                        author_parts.append(cleaned_p)
            
            auteur_clean = ", ".join(author_parts) if author_parts else ""

            # 2. Construction des composants du nom de fichier
            name_parts = [s_clean]

            # Tome / Numéro
            if t_clean:
                name_parts.append(t_clean)

            # Titre de l'album (si présent et différent du nom de la série)
            if a_csv_clean and a_csv_clean.lower().strip() != s_clean.lower().strip():
                name_parts.append(a_csv_clean)

            # Auteur (à la fin et séparé par un tiret)
            if auteur_clean:
                name_parts.append(auteur_clean)

            # Année (juste après, séparée par un tiret)
            if y_clean:
                name_parts.append(y_clean)

            # Joindre les composants avec " - " sans parenthèses ni crochets
            new_filename_base = " - ".join(name_parts)
            new_filename_base = " ".join(new_filename_base.split())

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
        
        # Le start_id du lot (1-indexed)
        start_id = start_idx + 1
        csv_response = call_gemini_api(batch_files, batch_num, total_batches, start_id)
        
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