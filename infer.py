# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Import dependencies

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:22:04.887542Z","iopub.execute_input":"2025-01-14T00:22:04.887853Z","iopub.status.idle":"2025-01-14T00:22:20.044239Z","shell.execute_reply.started":"2025-01-14T00:22:04.887830Z","shell.execute_reply":"2025-01-14T00:22:20.043323Z"},"jupyter":{"outputs_hidden":false}}
import json
import transformers
import torch

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from math import ceil
from datasets import Dataset
from transformers import pipeline
from itertools import chain

# Notify
print("Successfull import")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Load model

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:22:45.387156Z","iopub.execute_input":"2025-01-14T00:22:45.387759Z","iopub.status.idle":"2025-01-14T00:22:51.922089Z","shell.execute_reply.started":"2025-01-14T00:22:45.387731Z","shell.execute_reply":"2025-01-14T00:22:51.920833Z"},"jupyter":{"outputs_hidden":false}}
model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = pipeline("zero-shot-classification", model=model_name, device=device)

# Notify
print("Classifier loaded successfully")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Load GND taxonomy

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:22:56.005174Z","iopub.execute_input":"2025-01-14T00:22:56.005514Z","iopub.status.idle":"2025-01-14T00:22:58.572668Z","shell.execute_reply.started":"2025-01-14T00:22:56.005484Z","shell.execute_reply":"2025-01-14T00:22:58.571726Z"},"jupyter":{"outputs_hidden":false}}
# Load GND subject
gnd_taxonomy_path = "./GND-Subjects-all.json"

with open(gnd_taxonomy_path, "r", encoding="utf-8") as f:
    gnd_data = json.load(f)

# Notify
print("GND Taxonomy loaded successfully")

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:23:03.839604Z","iopub.execute_input":"2025-01-14T00:23:03.839888Z","iopub.status.idle":"2025-01-14T00:23:03.845979Z","shell.execute_reply.started":"2025-01-14T00:23:03.839866Z","shell.execute_reply":"2025-01-14T00:23:03.845229Z"},"jupyter":{"outputs_hidden":false}}
def get_gnd_code_from_name(name, gnd_list):
    for s in gnd_list:
        if(s['Name'] == name):
            return s['Code']

    return 0

# Notify
print("Function get_gnd_code_from_name is ready")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Building sublists for your gnd taxonomy (100 per sublist)

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:25:01.184347Z","iopub.execute_input":"2025-01-14T00:25:01.184658Z","iopub.status.idle":"2025-01-14T00:25:01.198443Z","shell.execute_reply.started":"2025-01-14T00:25:01.184634Z","shell.execute_reply":"2025-01-14T00:25:01.197355Z"},"jupyter":{"outputs_hidden":false}}
def subdivise_list(original_list, sublist_len):
    """
    Divides a list into sub-lists of specified size.
    
    :param list: The initial list containing objects.
    :param sub_list_size: The size of the sub-lists.
    :return: A list containing the sub-lists.
    """
    return [original_list[i:i + sublist_len] for i in range(0, len(original_list), sublist_len)]

# Subdivide in sublist of 100 items
gnd_sublists = subdivise_list(gnd_data, 100)
    
# Notify
print("taxonomy subdivided with success")

# %% [raw] {"jupyter":{"outputs_hidden":false}}
# 

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:25:05.131166Z","iopub.execute_input":"2025-01-14T00:25:05.131477Z","iopub.status.idle":"2025-01-14T00:25:05.136028Z","shell.execute_reply.started":"2025-01-14T00:25:05.131455Z","shell.execute_reply":"2025-01-14T00:25:05.135157Z"},"jupyter":{"outputs_hidden":false}}
print(len(gnd_sublists))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Building lists of candidates_labels

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:25:10.451034Z","iopub.execute_input":"2025-01-14T00:25:10.451352Z","iopub.status.idle":"2025-01-14T00:25:10.487656Z","shell.execute_reply.started":"2025-01-14T00:25:10.451328Z","shell.execute_reply":"2025-01-14T00:25:10.486905Z"},"jupyter":{"outputs_hidden":false}}
candidates_labels_list = [[item["Name"] for item in sublist] for sublist in gnd_sublists]

# Notify
print("Candidates_labels sublist are ready")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Load one file tibkak records

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:25:14.891554Z","iopub.execute_input":"2025-01-14T00:25:14.891839Z","iopub.status.idle":"2025-01-14T00:25:15.022237Z","shell.execute_reply.started":"2025-01-14T00:25:14.891818Z","shell.execute_reply":"2025-01-14T00:25:15.021428Z"},"jupyter":{"outputs_hidden":false}}
# Dataset name
dataset_name = "tib-core"

# Load tiblat record file
tibkat_record_file = f"./tibkat_test_{dataset_name}_subjects.json"

# Get list of parent class
with open(tibkat_record_file, "r", encoding="utf-8") as f:
    tibkat_record_list = json.load(f)

tibkat_dataset = Dataset.from_list(tibkat_record_list)

# Notify
print("All Tibkat records loaded successfully")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Run Predictions and save

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Define prediction function with optimisation techniques - 1

# %% [code] {"jupyter":{"outputs_hidden":false}}

# # Mise en cache des résultats pour accélérer les appels redondants
# @lru_cache(maxsize=None)
# def get_cached_gnd_code(label, current_gnd_sublist):
#     return get_gnd_code_from_name(label, current_gnd_sublist)

# # Fonction pour traiter un fichier et retourner ses prédictions sur GPU
# def process_item_on_gpu(item, candidates_labels_list, gnd_sublists, classifier, device):
#     filename = item['filename']
#     input_text = item['text']
#     predicted_labels = []

#     # Déplacer l'input vers le GPU
#     # input_text_tensor = torch.tensor(input_text).to(device)

#     # Traitement par lots
#     for current_group, current_gnd_sublist in zip(candidates_labels_list, gnd_sublists):
#         # Exécuter le classifieur sur le GPU
#         output = classifier(input_text, current_group, multi_label=True)

#         # Ajouter les prédictions au tableau
#         predicted_labels.extend(
#             {'code': get_cached_gnd_code(label, tuple(current_gnd_sublist)), 'score': score}
#             for label, score in zip(output['labels'], output['scores'])
#         )

#         print(f"{filename} treated")

#     # Trier les labels prédits par score décroissant
#     sorted_predicted_labels = sorted(predicted_labels, key=lambda x: x["score"], reverse=True)[:50]

#     # Retourner le résultat final pour ce fichier
#     return {
#         "filename": filename,
#         "predictions": sorted_predicted_labels
#     }

# # Fonction principale pour paralléliser le traitement avec les GPU
# def parallel_prediction_on_gpu(tibkat_record_list, candidates_labels_list, gnd_sublists, classifier, max_workers=4, device='cuda'):
#     # Initialisation du GPU
#     device = torch.device(device)

#     # Parallélisation avec ThreadPoolExecutor
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(
#             lambda item: process_item_on_gpu(item, candidates_labels_list, gnd_sublists, classifier, device),
#             tibkat_record_list
#         ))

#     return results


# # Notify
# print("Prediction function define with success")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Define prediction function with optimisation techniques - 2

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T00:18:19.540800Z","iopub.execute_input":"2025-01-14T00:18:19.541168Z","iopub.status.idle":"2025-01-14T00:18:19.553415Z","shell.execute_reply.started":"2025-01-14T00:18:19.541141Z","shell.execute_reply":"2025-01-14T00:18:19.552551Z"},"jupyter":{"outputs_hidden":false}}

# Mise en cache pour éviter les calculs redondants
@lru_cache(maxsize=None)
def get_cached_gnd_code(label, current_gnd_sublist):
    return get_gnd_code_from_name(label, current_gnd_sublist)

# Fonction pour traiter un batch sur un GPU donné
def process_batch_on_gpu(batch, candidates_labels_list, gnd_sublists, classifier, device):
    results = []

    # Préparer les inputs du batch
    batch_texts = [item['text'] for item in batch]
    batch_filenames = [item['filename'] for item in batch]

    # Initialiser un conteneur pour les prédictions par fichier
    batch_predictions = {filename: [] for filename in batch_filenames}

    group_index = 0

    # Itérer sur les groupes de labels
    for current_group, current_gnd_sublist in zip(candidates_labels_list, gnd_sublists):
        # Exécuter le classifieur sur tous les textes du batch pour le groupe courant
        output = classifier(batch_texts, current_group, multi_label=True)  # Batch complet

        # Processer les sorties par élément
        for idx, result in enumerate(output):
            filename = batch_filenames[idx]

            # Ajouter les prédictions pour chaque élément du batch
            batch_predictions[filename].extend(
                {'code': get_cached_gnd_code(label, tuple(current_gnd_sublist)), 'score': score}
                for label, score in zip(result['labels'], result['scores'])
            )

        print(f"{filename} - Group {group_index} treated")
        group_index = group_index + 1 

    # Trier et structurer les résultats pour chaque fichier
    for filename, predictions in batch_predictions.items():
        sorted_predicted_labels = sorted(predictions, key=lambda x: x["score"], reverse=True)[:50]
        results.append({
            "filename": filename,
            "predictions": sorted_predicted_labels
        })

    return results


# Fonction principale pour paralléliser le traitement avec batch et GPU
def parallel_prediction_2(
    tibkat_record_list, candidates_labels_list, gnd_sublists, model_name, batch_size=8, max_workers=4
):
    # Détecter tous les GPU disponibles
    available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    num_gpus = len(available_devices)
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    print(f"Using {num_gpus} GPUs for processing...")

    # Charger un pipeline par GPU
    classifiers = [
        pipeline("zero-shot-classification", model=model_name, device=device.index)
        for device in available_devices
    ]

    # Diviser les fichiers en batches
    num_batches = ceil(len(tibkat_record_list) / batch_size)
    batches = [tibkat_record_list[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, batch in enumerate(batches):
            device = available_devices[idx % num_gpus]  # Répartir les batches entre GPUs
            classifier = classifiers[idx % num_gpus]  # Utiliser le pipeline du bon GPU
            futures.append(
                executor.submit(
                    process_batch_on_gpu,
                    batch, candidates_labels_list, gnd_sublists, classifier, device
                )
            )

        # Récupérer les résultats
        for future in futures:
            results.extend(future.result())

    return results


# Notify
print("Batch-enabled multi-GPU prediction function defined successfully")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Define prediction function with optimisation techniques - 3

# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T01:11:08.824761Z","iopub.execute_input":"2025-01-14T01:11:08.825182Z","iopub.status.idle":"2025-01-14T01:11:08.835918Z","shell.execute_reply.started":"2025-01-14T01:11:08.825156Z","shell.execute_reply":"2025-01-14T01:11:08.834827Z"},"jupyter":{"outputs_hidden":false}}
def process_batch_with_multi_gpu(batch, candidates_labels_list, gnd_sublists, classifier, devices):
    """
    Traite un batch de fichiers en utilisant deux GPU.
    """
    results = []

    # Diviser les `candidates_labels_list` et `gnd_sublists` en deux groupes
    mid_point = len(candidates_labels_list) // 2
    group_1_labels, group_2_labels = candidates_labels_list[:mid_point], candidates_labels_list[mid_point:]
    group_1_gnd, group_2_gnd = gnd_sublists[:mid_point], gnd_sublists[mid_point:]

    def process_on_device(device, batch, groups, gnd_subs):
        """
        Fonction pour traiter un lot de données sur un GPU donné.
        """
        batch_predictions = []
        for item in batch:
            filename = item['filename']
            input_text = item['text']
            predicted_labels = []
    
            # Itérer sur les groupes pour ce GPU
            for current_group, current_gnd_sublist in zip(groups, gnd_subs):
                # Exécuter le classifieur sur le GPU
                outputs = classifier([input_text], current_group, multi_label=True)
    
                # Vérifier si `outputs` est une liste
                if isinstance(outputs, list):
                    for output in outputs:  # Parcourir chaque dict dans la liste
                        predicted_labels.extend(
                            {'code': get_gnd_code_from_name(label, tuple(current_gnd_sublist)), 'score': score}
                            for label, score in zip(output['labels'], output['scores'])
                        )
                else:
                    raise ValueError(f"Unexpected output format: {outputs}")
    
            # Trier par score décroissant
            sorted_predicted_labels = sorted(predicted_labels, key=lambda x: x["score"], reverse=True)[:50]
    
            # Ajouter le résultat pour ce fichier
            batch_predictions.append({
                "filename": filename,
                "predictions": sorted_predicted_labels
            })
    
            print(f"{filename} treated on device {device}")
    
        return batch_predictions


    # Lancer le traitement sur deux GPU en parallèle
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_1 = executor.submit(process_on_device, devices[0], batch, group_1_labels, group_1_gnd)
        future_2 = executor.submit(process_on_device, devices[1], batch, group_2_labels, group_2_gnd)

        # Récupérer les résultats
        results.extend(future_1.result())
        results.extend(future_2.result())

    return results


# Fonction principale pour le traitement
def parallel_prediction_3(tibkat_record_list, candidates_labels_list, gnd_sublists, classifier, batch_size=8):
    """
    Traite les fichiers par lots en utilisant deux GPU.
    """
    all_predictions = []

    # Diviser les fichiers en batches
    for i in range(0, len(tibkat_record_list), batch_size):
        batch = tibkat_record_list[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(tibkat_record_list) + batch_size - 1) // batch_size}...")

        # Lancer le traitement sur deux GPU
        batch_predictions = process_batch_with_multi_gpu(
            batch, candidates_labels_list, gnd_sublists, classifier, devices=['cuda:0', 'cuda:1']
        )

        # Ajouter les prédictions au résultat global
        all_predictions.extend(batch_predictions)

    return all_predictions

# Notify
print("Optimization technique 3 defined")

# %% [markdown]
# ### Define prediction function with optimisation techniques - 4

# %% [code]
def batch_predict(batch, candidates_labels_list, gnd_sublists, classifier):
    """
    Predicts labels for a batch of texts using the classifier.
    """
    predictions = []

    for current_group, current_gnd_sublist in zip(candidates_labels_list, gnd_sublists):
        # Perform classification for the current group of labels
        output = classifier(batch["text"], current_group, multi_label=True)

        # Check if output is a list of results
        if isinstance(output, list):
            for single_output in output:
                group_predictions = [
                    {
                        "code": get_gnd_code_from_name(label, current_gnd_sublist),
                        "label": label,
                        "score": score,
                    }
                    for label, score in zip(single_output["labels"], single_output["scores"])
                ]
                # Sort predictions by score in descending order and take the top 50
                group_predictions.sort(key=lambda x: x["score"], reverse=True)
                predictions.extend(group_predictions[:50])
        else:
            group_predictions = [
                {
                    "code": get_gnd_code_from_name(label, current_gnd_sublist),
                    "label": label,
                    "score": score,
                }
                for label, score in zip(output["labels"], output["scores"])
            ]
            # Sort predictions by score in descending order and take the top 50
            group_predictions.sort(key=lambda x: x["score"], reverse=True)
            predictions.extend(group_predictions[:50])

    batch["predictions"] = predictions

    # Debugging information
    print(f"Batch predictions:\n{batch}")

    return batch


def parallel_prediction_4(dataset, batch_size, candidates_labels_list, gnd_sublists, classifier):
    """
    Processes a dataset in batches and returns predictions.
    Automatically distributes batches across multiple GPUs if available.
    """
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for inference.")

    # Split dataset into chunks for each GPU
    def process_on_device(device, data_subset):
        classifier.device = torch.device(f"cuda:{device}")
        return data_subset.map(
            batch_predict,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                'candidates_labels_list': candidates_labels_list,
                'gnd_sublists': gnd_sublists,
                'classifier': classifier
            }
        )

    if num_gpus > 1:
        dataset_splits = dataset.shard(num_shards=num_gpus, index=0, contiguous=True)
        results = []
        for gpu_id in range(num_gpus):
            split_result = process_on_device(gpu_id, dataset_splits.select(range(gpu_id, len(dataset_splits), num_gpus)))
            results.append(split_result)

        # Combine results from all GPUs
        all_results = torch.utils.data.ConcatDataset(results)
    else:
        all_results = dataset.map(
            batch_predict,
            batched=True,
            batch_size=batch_size,
            fn_kwargs={
                'candidates_labels_list': candidates_labels_list,
                'gnd_sublists': gnd_sublists,
                'classifier': classifier
            }
        )

    return all_results

# Notify
print("Optimization technique 4 defined")


# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Run predictions

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2025-01-14T01:11:14.791899Z","iopub.execute_input":"2025-01-14T01:11:14.792288Z","execution_failed":"2025-01-14T10:45:59.626Z"},"jupyter":{"outputs_hidden":false}}
# output list
# all_predictions = []

# for item in tibkat_record_list:
#     filename = item['filename']
#     input_text = item['text']
#     predicted_labels = []

#     for i in range(len(candidates_labels_list)) :
#         current_group = candidates_labels_list[i]
#         current_gnd_sublist = gnd_sublists[i]
#         output = classifier(input_text, current_group, multi_label=True)

#         predicted_labels.append({
#             'code': get_gnd_code_from_name(label, current_gnd_sublist), 
#             'score': score
#         }
#                 for label, score in zip(output['labels'], output['scores'])
#         )

#         print(f"{filename} - Group {i} treaded")

#     # at this point, predicted_labels must be a array 204739 items
#     sorted_predicted_labels = sorted(predicted_labels, key=lambda x: x["score"], reverse=True)

#     # save prediction for this file
#     all_predictions.append({
#         "filename": filename,
#         "predictions": sorted_predicted_labels[:50]
#     })

#     print(f"{filename} treated")

# Appel technique 1 ------------------------------------
# all_predictions = parallel_prediction_4(
#     dataset=tibkat_record_list,
#     candidates_labels_list=candidates_labels_list,
#     gnd_sublists=gnd_sublists,
#     classifier=classifier,
#     max_workers=8,  # Ajuste selon les cœurs du CPU et le nombre de GPU
#     device='cuda:0'  # Utilise le premier GPU. 'cuda:1' pour le deuxième GPU
# )

# Appel technique 2 ------------------------------------
all_predictions = parallel_prediction_2(
    tibkat_record_list=tibkat_record_list,
    candidates_labels_list=candidates_labels_list,
    gnd_sublists=gnd_sublists,
    model_name=model_name,
    batch_size=8,
    max_workers=2
)

# Appel technique 3 ------------------------------------
# all_predictions = parallel_prediction_3(
#     tibkat_record_list=tibkat_record_list,
#     candidates_labels_list=candidates_labels_list,
#     gnd_sublists=gnd_sublists,
#     classifier=classifier,
#     batch_size=8
# )

# Appel technique 4 ------------------------------------
# all_predictions = parallel_prediction_4(
#     dataset=tibkat_dataset,
#     candidates_labels_list=candidates_labels_list,
#     gnd_sublists=gnd_sublists,
#     classifier=classifier,
#     batch_size=8
# )


# Save predictions
from datetime import date

# get current date to tag output predction file
today = date.today()

prediction_file = f"./dz_{dataset_name}_predictions_{today}.json"
with open(prediction_file, "w", encoding="utf-8") as f:
    json.dump(all_predictions, f, indent=4, ensure_ascii=False)


# Notify
print("Annotation ended successfully")