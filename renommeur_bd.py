import os
import shutil
import csv
import io
import math
import threading
import time
from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# Utilisation du client Google GenAI
from google import genai
from google.genai import types

# --- Configuration ---
SOURCE_DIR = os.environ.get("SOURCE_DIR", r"F:\Téléchargements\_BD\titi")
DEST_DIR = os.environ.get("DEST_DIR", r"F:\Téléchargements\_BD\titi-renommes")

DRY_RUN = os.environ.get("DRY_RUN", "False").lower() in ("true", "1", "yes")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

# Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Local LLM (ex: LM Studio)
LOCAL_API_URL = os.environ.get("LOCAL_API_URL", "http://localhost:1234/v1")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "local-model")

extensions_raw = os.environ.get("VALID_EXTENSIONS", ".cbz,.cbr,.pdf,.epub")
VALID_EXTENSIONS = tuple(ext.strip().lower() for ext in extensions_raw.split(",") if ext.strip())
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 25))

# --- Prompt système pour l'extraction des métadonnées ---
SYSTEM_INSTRUCTION = """Tu reçois une liste numérotée de fichiers de BD (sous la forme "ID. Nom_de_fichier").
Ta tâche est de générer le nouveau nom de fichier idéal pour chaque BD sous forme de tableau CSV à deux colonnes : "ID";"Nouveau_Nom".

Ne renvoie aucun texte d'explication ou de salutation. Retourne uniquement le bloc de code CSV (entouré de ```csv et ```).

Règles pour construire "Nouveau_Nom" :
1. Format cible standard : Nom de la série - Numéro du tome - Nom du tome - Auteur - Année.
   - Si un ou plusieurs de ces éléments (tome, auteur, année) sont inconnus, absents ou vides, omets-les complètement.
   - N'insère jamais de séparateurs consécutifs (comme ` - - - ` ou ` - - `) ni de séparateur inutile. Le nom final doit être propre, par exemple : `Série - Auteur - Année` ou `Série - Numéro - Titre - Auteur`.
2. Pas de valeurs fictives ou placeholder : N'utilise JAMAIS de valeurs génériques telles que "Inconnu", "Unknown", "NA", "N/A", "None", "Auteur", "Année", "Pas d'auteur", ou similaire pour combler un champ manquant. Si un élément est inconnu, omets-le simplement.
3. Éviter les répétitions : Si le nom du tome est identique au nom de la série (ou s'il n'y a pas de titre de tome distinct), n'écris pas le nom du tome (ex: écris `Série - 01 - Auteur - Année` et non `Série - 01 - Série - Auteur - Année`).
4. Reconstitution des articles inversés et corrections :
   - Si le nom contient un article inversé (souvent placé à la fin entre parenthèses, ex: "Brève Histoire de LAvenir (Une)"), remets-le naturellement au début du titre (ex: "Une Brève Histoire de L'Avenir").
   - Restaure les apostrophes ou accents manquants (ex: "LAvenir" devient "L'Avenir").
5. Séparateur : Utilise uniquement un tiret entouré d'espaces ( - ) pour séparer les éléments présents.
6. Pas de symboles de démarcation : Ne mets pas de crochets [ ] ni de parenthèses ( ) autour de l'auteur, de l'année ou du tome pour les isoler (par exemple, écris `Auteur - Année` et non `[Auteur] - (Année)`). Cependant, si des parenthèses ou crochets font partie intégrante du titre d'une série ou d'une BD d'origine (ex: "Squeak The Mouse (FR)"), conserve-les.
7. Uniformisation du Tome :
   - Pour les tomes standards, le numéro de tome doit être composé uniquement de chiffres, précédé d'un zéro s'il n'y a qu'un chiffre (ex: "01", "05", "12"). Ne mets pas de préfixe comme "T01" ou "Tome 01".
   - Pour les intégrales ou les tomes spéciaux, conserve la structure d'origine (ex: "INT1" ou "INT2" doivent rester "INT1" ou "INT2", ne les convertis pas en simples chiffres).
   - S'il n'y a pas de numéro de tome (One Shot ou intégrale unique sans numéro), laisse ce champ vide.
8. Complétion par IA : Utilise tes connaissances générales sur les bandes dessinées pour ajouter l'auteur (ou dessinateur), l'année de publication ou corriger le nom de l'album s'il est incomplet ou erroné dans le nom d'origine.
9. Extension : Conserve l'extension d'origine du fichier (ex: .cbz, .cbr, .pdf).
"""

def run_with_interrupt_protection(func, *args, **kwargs):
    """Exécute une fonction bloquante dans un thread en permettant au thread principal de capter instantanément le Ctrl+C sous Windows."""
    result = [None]
    error = [None]
    finished = threading.Event()
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            error[0] = e
        finally:
            finished.set()
            
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    
    # Boucle de sommeil courte pour permettre la livraison du signal SIGINT sous Windows
    while not finished.is_set():
        time.sleep(0.1)
        
    if error[0]:
        raise error[0]
    return result[0]

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

def call_llm_api(filenames, batch_num=1, total_batches=1, start_id=1):
    if not filenames:
        print("Aucun fichier à envoyer à l'API.")
        return None

    # Numéroter les fichiers pour le prompt de l'API (ex: "1. Fichier.cbz")
    payload_lines = [f"{start_id + i}. {filename}" for i, filename in enumerate(filenames)]
    payload_content = "\n".join(payload_lines)
    
    print(f"\n=== Lot {batch_num}/{total_batches} : {len(filenames)} fichiers ===")
    
    # TRACE : Afficher la liste des fichiers du lot
    print(f"📋 Fichiers envoyés à l'API ({LLM_PROVIDER}) :")
    for i, filename in enumerate(filenames):
        print(f"  {start_id + i:2d}. {filename}")
    print("-" * 80)
    
    csv_content = None

    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
            print("ERREUR: La clé API Gemini n'est pas configurée dans le fichier .env.")
            return None
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            print(f"🚀 Envoi du lot {batch_num} à l'API Gemini...")
            print("✍️ Réponse du LLM au fur et à mesure :")
            print("-" * 40)
            
            response = client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=payload_content,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                )
            )
            full_text = []
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_text.append(chunk.text)
            print()
            print("-" * 40)
            csv_content = "".join(full_text)
        except Exception as e:
            print(f"❌ ERREUR lors de l'appel à l'API Gemini pour le lot {batch_num}: {e}")
            return None
            
    elif LLM_PROVIDER == "local":
        import httpx
        url = f"{LOCAL_API_URL.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": LOCAL_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": payload_content}
            ],
            "temperature": 0.1,
            "stream": True
        }
        try:
            print(f"🚀 Envoi du lot {batch_num} à l'API locale (LM Studio / {LOCAL_MODEL})...")
            print("✍️ Réponse du LLM au fur et à mesure :")
            print("-" * 40)
            
            full_text = []
            with httpx.stream("POST", url, json=data, headers=headers, timeout=120.0) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[len("data: "):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            chunk_json = json.loads(data_str)
                            if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                delta = chunk_json["choices"][0].get("delta", {})
                                reasoning = delta.get("reasoning_content", "")
                                content = delta.get("content", "")
                                if reasoning:
                                    print(reasoning, end="", flush=True)
                                if content:
                                    print(content, end="", flush=True)
                                    full_text.append(content)
                        except Exception:
                            pass
            print()
            print("-" * 40)
            csv_content = "".join(full_text)
        except Exception as e:
            print(f"❌ ERREUR lors de l'appel à l'API locale pour le lot {batch_num}: {e}")
            return None
    else:
        print(f"ERREUR: Fournisseur LLM inconnu '{LLM_PROVIDER}'. Modifiez votre fichier .env.")
        return None

    if not csv_content:
        print("ERREUR: Réponse vide de l'API.")
        return None

    # Nettoyage du CSV
    if isinstance(csv_content, str):
        import re
        # Supprimer le bloc de réflexion <think>...</think> s'il a été inclus dans le contenu textuel
        csv_content = re.sub(r'<think>.*?</think>', '', csv_content, flags=re.DOTALL)
        csv_content = csv_content.strip()
        
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
            
            if cleaned_row and 'ID' in cleaned_row and 'Nouveau_Nom' in cleaned_row:
                parsed_data.append(cleaned_row)
                
    except Exception as e:
        print(f"ERREUR lors du parsing du CSV du lot {batch_num}: {e}")
        return []
    
    print(f"Lot {batch_num}: {len(parsed_data)} entrées parsées.")
    return parsed_data

def sanitize_filename(filename):
    """Supprime les caractères invalides pour les noms de fichiers Windows."""
    if not filename:
        return ""
    invalid_chars = r'<>:"/\|?*' + "".join(chr(i) for i in range(32))
    sanitized = "".join(c for c in filename if c not in invalid_chars)
    return sanitized.strip()

def process_files(all_parsed_data, original_files_in_source, start_id=1):
    """Copie et renomme les fichiers BD dans le dossier de résultat (ou simule le renommage)."""
    if not all_parsed_data:
        print("Aucune donnée parsée à traiter.")
        return 0, 0

    # Créer le dossier de destination si on n'est pas en mode simulation
    if not DRY_RUN:
        if not os.path.exists(DEST_DIR):
            try:
                os.makedirs(DEST_DIR)
                print(f"📁 Dossier de destination '{DEST_DIR}' créé.")
            except OSError as e:
                print(f"ERREUR: Impossible de créer le dossier de destination '{DEST_DIR}': {e}")
                return 0, 0

    processed_count = 0
    error_count = 0
    
    # Associer les métadonnées aux fichiers via l'ID (index dans original_files_in_source)
    found_matches = {}
    missing_in_csv = []
    processed_indices = set()
    
    for data_item in all_parsed_data:
        id_str = data_item.get('ID', '')
        new_filename = data_item.get('Nouveau_Nom', '')
        try:
            # 1-indexed dans le CSV -> index relatif dans le lot actuel
            file_idx = int(id_str) - start_id
            if 0 <= file_idx < len(original_files_in_source) and new_filename:
                source_file = original_files_in_source[file_idx]
                found_matches[source_file] = new_filename
                processed_indices.add(file_idx)
        except (ValueError, TypeError):
            pass
            
    # Identifier les fichiers non traités
    for idx, source_file in enumerate(original_files_in_source):
        if idx not in processed_indices:
            missing_in_csv.append(source_file)

    # Préparer les données pour la table d'affichage
    table_rows = []

    for original_filename_from_fs in original_files_in_source:
        if original_filename_from_fs not in found_matches:
            table_rows.append((original_filename_from_fs, "N/A", "IGNORÉ"))
            error_count += 1
            continue
        
        new_filename = found_matches[original_filename_from_fs]
        original_full_path = os.path.join(SOURCE_DIR, original_filename_from_fs)
        
        try:
            # Nettoyer le nom de fichier final pour Windows
            new_filename_clean = sanitize_filename(new_filename)
            
            if not new_filename_clean or new_filename_clean == os.path.splitext(original_filename_from_fs)[1]:
                table_rows.append((original_filename_from_fs, new_filename_clean or "Nom vide", "ERREUR"))
                error_count += 1
                continue

            dest_full_path = os.path.join(DEST_DIR, new_filename_clean)

            if DRY_RUN:
                table_rows.append((original_filename_from_fs, new_filename_clean, "SIMULÉ"))
                processed_count += 1
            else:
                shutil.copy2(original_full_path, dest_full_path)
                table_rows.append((original_filename_from_fs, new_filename_clean, "OK"))
                processed_count += 1

        except Exception as e:
            table_rows.append((original_filename_from_fs, new_filename, "ERREUR"))
            error_count += 1

    # --- AFFICHAGE DE LA TABLE DES RÉSULTATS ---
    # Calcul dynamique des largeurs de colonnes pour éviter de couper les noms de fichiers
    w_orig = max((len(orig) for orig, _, _ in table_rows), default=12)
    w_new = max((len(new) for _, new, _ in table_rows), default=11)
    
    # S'assurer d'avoir au moins la largeur des titres de colonnes
    w_orig = max(w_orig, 12)  # "Nom Original"
    w_new = max(w_new, 11)    # "Nom Proposé"
    w_status = 9              # "Statut"
    
    # Entête
    header = f"| {'Nom Original':<{w_orig}} | {'Nom Proposé':<{w_new}} | {'Statut':<{w_status}} |"
    separator = "+" + "-" * (w_orig + 2) + "+" + "-" * (w_new + 2) + "+" + "-" * (w_status + 2) + "+"
    
    print("\n📊 RÉSULTATS DU TRAITEMENT :")
    if DRY_RUN:
        print("⚠️  [MODE SIMULATION ACTIVE] Aucun fichier n'a été réellement créé ou renommé.")
    
    print(separator)
    print(header)
    print(separator)
    
    for orig, new, status in table_rows:
        print(f"| {orig:<{w_orig}} | {new:<{w_new}} | {status:<{w_status}} |")
        
    print(separator)
    
    return processed_count, error_count

# --- Exécution principale avec pagination ---
if __name__ == "__main__":
    try:
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
        
        total_processed = 0
        total_errors = 0
        
        # Traiter par lots
        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            batch_files = bd_filenames[start_idx:end_idx]
            
            print(f"\n🔄 Traitement du lot {batch_num}/{total_batches}")
            
            # Le start_id du lot (1-indexed)
            start_id = start_idx + 1
            csv_response = run_with_interrupt_protection(call_llm_api, batch_files, batch_num, total_batches, start_id)
            
            if csv_response:
                parsed_data = parse_csv_response(csv_response, batch_num)
                if parsed_data:
                    # Traiter et afficher le tableau des résultats immédiatement pour ce lot
                    processed_count, error_count = process_files(parsed_data, batch_files, start_id)
                    total_processed += processed_count
                    total_errors += error_count
                else:
                    print(f"❌ Échec du parsing du CSV du lot {batch_num}")
                    total_errors += len(batch_files)
            else:
                print(f"❌ Échec du traitement du lot {batch_num}")
                total_errors += len(batch_files)
        
        # --- RÉSULTATS FINAUX GLOBAUX ---
        print(f"\n🎯 === RÉSULTATS FINAUX GLOBAUX ===")
        if DRY_RUN:
            print(f"✅ {total_processed} fichiers simulés avec succès")
        else:
            print(f"✅ {total_processed} fichiers copiés et renommés avec succès")
            
        print(f"❌ {total_errors} erreurs ou fichiers ignorés/non traités")
        if not DRY_RUN and total_processed > 0:
            print(f"📂 Fichiers disponibles dans: {DEST_DIR}")

        print("\n--- Fin du script ---")
    except KeyboardInterrupt:
        print("\n\n🛑 Interruption par l'utilisateur (Ctrl+C). Arrêt immédiat.")
        os._exit(0)