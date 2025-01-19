import json
import torch_xla
import torch_xla.core.xla_model as xm
import os
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from datasets import Dataset
from transformers import pipeline

# Définir l'appareil TPU
device = xm.xla_device()

# Charger le modèle sur TPU
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
classifier = pipeline("zero-shot-classification", model=model_name, device=device)

print("Classifier chargé avec succès sur TPU")

# function pour uniquement les fichier non traiter
def filter_unprocessed_records(tibkat_record_list, predictions_folder, last_processed_filename=None):
    """
    Filtre la liste des documents pour ne conserver que ceux qui n'ont pas encore été traités.

    Args:
        tibkat_record_list (list): Liste des enregistrements Tibkat.
        predictions_folder (str): Chemin vers le dossier contenant les fichiers de prédiction.
        last_processed_filename (str, optional): Nom du dernier fichier traité. Si fourni, 
                                                 filtre à partir de ce fichier.

    Returns:
        list: Liste des enregistrements non traités.
    """
    # Charger les fichiers déjà traités dans le dossier de prédictions
    processed_files = set()
    if not last_processed_filename:
        if os.path.exists(predictions_folder):
            for file in os.listdir(predictions_folder):
                if file.endswith(".json"):
                    processed_files.add(file)
        else:
            return tibkat_record_list
    else:
        # Si un fichier est spécifié, trouver sa position dans la liste
        last_processed_index = None
        for i, record in enumerate(tibkat_record_list):
            if record["filename"] == last_processed_filename:
                last_processed_index = i
                break
        if last_processed_index is not None:
            # Retourner uniquement les fichiers après le dernier traité
            return tibkat_record_list[last_processed_index + 1 :]
        else:
            print(f"Warning: Filename {last_processed_filename} not found in tibkat_record_list.")
            return []

    # Filtrer les fichiers non encore traités
    unprocessed_records = [
        record for record in tibkat_record_list if f"{record['filename']}.json" not in processed_files
    ]

    return unprocessed_records


def get_gnd_code_from_name(name, gnd_list):
    for s in gnd_list:
        if(s['Name'] == name):
            return s['Code']

    return 0

def subdivise_list(original_list, sublist_len):
    """
    Divides a list into sub-lists of specified size.
    
    :param list: The initial list containing objects.
    :param sub_list_size: The size of the sub-lists.
    :return: A list containing the sub-lists.
    """
    return [original_list[i:i + sublist_len] for i in range(0, len(original_list), sublist_len)]


# Fonction pour traiter un batch sur TPU
def process_batch_on_tpu(batch, candidates_labels_list, gnd_sublists, classifier):
    results = []
    batch_texts = [item['text'] for item in batch]
    batch_filenames = [item['filename'] for item in batch]

    group_index = 0
    for current_group, current_gnd_sublist in zip(candidates_labels_list, gnd_sublists):
        # Exécuter le classifieur sur TPU
        output = classifier(batch_texts, current_group, multi_label=True)
        for idx, result in enumerate(output):
            filename = batch_filenames[idx]
            results.append({
                "filename": filename,
                "predictions": [
                    {'code': get_gnd_code_from_name(label, current_gnd_sublist), 'score': score}
                    for label, score in zip(result['labels'], result['scores'])
                ]
            })
        print(f"{filename} - Groupe {group_index} traité")
        group_index += 1
    return results

# Fonction principale pour paralléliser le traitement
def parallel_prediction_tpu(
    tibkat_record_list, candidates_labels_list, gnd_sublists, model_name, batch_size=8, max_workers=4
):
    num_batches = ceil(len(tibkat_record_list) / batch_size)
    batches = [tibkat_record_list[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in batches:
            futures.append(
                executor.submit(
                    process_batch_on_tpu,
                    batch, candidates_labels_list, gnd_sublists, classifier
                )
            )
        for future in futures:
            results.extend(future.result())
    return results

dataset_name = "tib-core"
predictions_folder = "./predictions"

# Charger les données et exécuter
gnd_taxonomy_path = "./GND-Subjects-all.json"
with open(gnd_taxonomy_path, "r", encoding="utf-8") as f:
    gnd_data = json.load(f)

gnd_sublists = subdivise_list(gnd_data, 100)
candidates_labels_list = [[item["Name"] for item in sublist] for sublist in gnd_sublists]

# Load tiblat record file
tibkat_record_file = f"./tibkat_test_{dataset_name}_subjects.json"

# Get list of parent class
with open(tibkat_record_file, "r", encoding="utf-8") as f:
    tibkat_record_list = json.load(f)

# filter input list if we have checkpoint
tibkat_record_list = unprocessed_records = filter_unprocessed_records(tibkat_record_list, predictions_folder)

tibkat_dataset = Dataset.from_list(tibkat_record_list)

# Notify
print("All unprocessed files have been loaded")

# Exécuter les prédictions
all_predictions = parallel_prediction_tpu(
    tibkat_record_list=tibkat_record_list,
    candidates_labels_list=candidates_labels_list,
    gnd_sublists=gnd_sublists,
    model_name=model_name,
    batch_size=8,
    max_workers=4
)

# Sauvegarder les prédictions
from datetime import date
today = date.today()
output_file = f"./predictions_tpu_{today}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_predictions, f, indent=4, ensure_ascii=False)

print("Traitement terminé avec succès sur TPU")
