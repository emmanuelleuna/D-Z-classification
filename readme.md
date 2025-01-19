# Direct Zeroshot Classification
Ce script, contenu dans le fichier infer.py, exécute une classification zéro-shot sur des données textuelles à l'aide du modèle mDeBERTa-v3-base-mnli-xnli. Il utilise une taxonomie GND (Gemeinsame Normdatei) pour identifier les codes pertinents associés à chaque texte et génère des prédictions en utilisant des techniques d'optimisation sur GPU.

## Instructions pour l'exécution
### Dépendances : Assurez-vous que les bibliothèques suivantes sont installées :

torch
transformers
datasets  

Vous pouvez les installer avec la commande suivante :  

```pip install torch transformers datasets```  

Fichiers requis :  

- GND-Subjects-all.json : La taxonomie GND contenant les codes et les noms des sujets.
- tibkat_test_tib-core_subjects.json : Le fichier contenant les données textuelles à traiter.  

Placez ces fichiers dans le même répertoire que infer.py.  

Exécution : Lancez le script principal :  

`python infer.py`  

Paramètres de sortie : Le script génère un fichier de prédictions dans le répertoire courant. Ce fichier est nommé au format :

- dz_tib-core_predictions_<date>.json  
- Par exemple : dz_tib-core_predictions_2025-01-19.json.  

### Résultat  
Le fichier de sortie est un fichier JSON contenant les prédictions pour chaque texte du fichier tibkat_test_tib-core_subjects.json. Chaque entrée du fichier JSON inclut :

- filename : Le nom du fichier correspondant.  
- predictions : Une liste des 50 étiquettes les plus probables, avec :  
    code : Le code GND associé à l'étiquette.  
    label : Le libellé de l'étiquette.  
    score : La probabilité associée.  

### Utilisation des GPU  
Le script détecte automatiquement les GPUs disponibles et répartit les calculs entre eux. Si plusieurs GPUs sont disponibles, les données sont traitées en parallèle pour une meilleure performance.