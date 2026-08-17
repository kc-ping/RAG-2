# Leçon complète : RAG avec FAISS, LangChain et OpenAI

## 1) Objectif du projet

Ce projet permet de :

- lire des fichiers texte dans un dossier `docs/`
- découper ces textes en morceaux appelés chunks
- créer des embeddings avec OpenAI
- sauvegarder ces embeddings dans une base vectorielle locale
- retrouver les meilleurs documents quand on pose une question

Cela s'appelle un système RAG (Retrieval Augmented Generation).

Le but est simple : au lieu de donner toute une grande base de texte à l'IA d'un coup, on ne lui donne que les morceaux les plus pertinents pour la question posée.

---

## 2) Ce que fait le projet dans la vie réelle

Le projet simule une mini base de connaissances sur des entreprises comme :

- Google
- Microsoft
- Nvidia
- SpaceX
- Tesla

On lit ces fichiers texte, on les transforme en vecteurs, puis on pose une question comme :

> "What was NVIDIA's first graphics accelerator called?"

Le système recherche dans les textes les passages qui ressemblent le plus à la question et renvoie les meilleurs extraits.

---

## 3) Les fichiers du projet

### 3.1 `ingestion_pipeline.py`

C’est le script qui alimente la base de connaissances.

Il fait ceci :

1. vérifie que le dossier `docs/` existe
2. charge tous les fichiers `.txt`
3. les lit en UTF-8
4. découpe les textes en morceaux
5. génère les embeddings
6. sauvegarde la base vectorielle locale

### 3.2 `retrieval_pipeline.py`

C’est le script qui répond aux questions.

Il fait ceci :

1. vérifie que la base FAISS existe bien
2. charge la base sauvegardée
3. crée un retriever
4. envoie la question
5. récupère les meilleurs passages
6. affiche le contexte trouvé

### 3.3 `docs/`

Contient les textes sources. Ce sont les documents qui servent à fabriquer la base de connaissances.

---

## 4) Les technologies utilisées

### 4.1 Python

Python est le langage principal du projet. Il permet :

- lire des fichiers
- manipuler des textes
- appeler des API OpenAI
- utiliser des bibliothèques de traitement de données

### 4.2 LangChain

LangChain est utilisé pour :

- charger les documents
- découper les textes
- gérer les embeddings
- manipuler la base vectorielle

Les éléments importants sont :

- `DirectoryLoader` : charge plusieurs fichiers
- `TextLoader` : lit un fichier texte
- `CharacterTextSplitter` : découpe les documents en morceaux
- `OpenAIEmbeddings` : transforme le texte en vecteurs
- `FAISS` : base vectorielle locale pour la recherche sémantique

### 4.3 OpenAI Embeddings

Les embeddings sont des représentations numériques du texte.

Exemple :

- une phrase est transformée en un vecteur
- deux phrases proches sémantiquement ont des vecteurs proches
- le système peut donc comparer les questions avec les textes de manière intelligente

### 4.4 FAISS

FAISS est une base vectorielle rapide et locale.

Elle permet de stocker des embeddings et de retrouver rapidement les passages les plus proches d’une question.

---

## 5) Qu’est-ce qu’un RAG ?

RAG = Retrieval Augmented Generation.

La logique est la suivante :

1. l’utilisateur pose une question
2. le système cherche les passages les plus pertinents dans la base de connaissances
3. on donne ces passages à un modèle de langage comme contexte
4. le modèle répond à la question en s’appuyant sur ces informations

Dans ce projet, on a surtout la partie retrieval : recherche des passages utiles.

---

## 6) Les problèmes rencontrés et leurs solutions

### 6.1 Problème 1 : erreur de décodage Unicode

Le message d’erreur était du type :

> UnicodeDecodeError

Cela arrivait parce que les fichiers `.txt` étaient lus avec le bon encodage par défaut du système Windows, qui n’était pas toujours UTF-8.

#### Solution

On a ajouté explicitement :

```python
loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
```

Cela force Python à lire les fichiers en UTF-8, ce qui est le standard le plus sûr pour des textes modernes.

#### Pourquoi c’est important

Les fichiers de texte peuvent contenir :

- accents
- apostrophes
- caractères spéciaux
- symboles du monde entier

Si le bon encodage n’est pas utilisé, le texte est mal lu.

---

### 6.2 Problème 2 : base de données vide ou incomplète

Parfois, le dossier `db/` existait, mais il ne contenait pas de données réelles. Le script semblait croire que la base était prête alors qu’elle ne l’était pas.

#### Solution

On a vérifié la présence d’un vrai fichier de stockage :

```python
os.path.exists(os.path.join(persistent_directory, "index.faiss"))
```

Au lieu de supposer que le dossier signifie que la base existe, on vérifie directement l’index FAISS.

---

### 6.3 Problème 3 : Chroma crashait sur Windows

Le projet a d’abord utilisé Chroma, mais cette base vectorielle a provoqué des erreurs sur Windows, notamment des problèmes de dépendances natives et de backend.

#### Solution

Le projet a été migré vers FAISS, qui est plus stable pour un usage local et plus simple à déployer dans un environnement de développement.

C’est une étape clé du projet : le problème n’était pas le code RAG en lui-même, mais le moteur vectoriel choisi.

---

## 7) Explication du code de `ingestion_pipeline.py`

### 7.1 Importations

```python
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
```

#### Ce que cela signifie

- `os` : gestion des dossiers, fichiers, chemins
- `TextLoader` et `DirectoryLoader` : chargement des fichiers textes
- `CharacterTextSplitter` : découpage du texte
- `OpenAIEmbeddings` : génération des embeddings
- `FAISS` : stockage de vecteurs
- `load_dotenv` : charge les variables d’environnement comme la clé API OpenAI

---

### 7.2 `load_dotenv()`

```python
load_dotenv()
```

Cette ligne lit le fichier `.env` qui contient les clés secrètes comme la clé OpenAI.

Sans cela, le script ne pourrait pas appeler l’API d’embedding.

---

### 7.3 `load_documents(docs_path="docs")`

```python
def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")
```

Cette fonction vérifie si le dossier `docs/` existe et charge les fichiers `.txt`.

Le cœur est celui-ci :

```python
loader = DirectoryLoader(
    path=docs_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True}
)
```

#### Explication

- `path=docs_path` : chemin du dossier source
- `glob="*.txt"` : ne charge que les fichiers texte
- `loader_cls=TextLoader` : Lit les fichiers texte
- `encoding="utf-8"` : force l’encodage correct
- `autodetect_encoding=True` : détecte l’encodage si nécessaire

Ensuite :

```python
documents = loader.load()
```

Cela retourne une liste de documents. Chaque document contient :

- le contenu texte
- les métadonnées (nom du fichier, source, etc.)

---

### 7.4 `split_documents(documents, chunk_size=1000, chunk_overlap=0)`

```python
def split_documents(documents, chunk_size=1000, chunk_overlap=0):
```

Cette fonction découpe les gros textes en petits morceaux plus faciles à comparer.

```python
text_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
```

#### Pourquoi couper les textes ?

Parce que :

- un gros document est trop long pour être traité efficacement
- les embeddings tiennent mieux sur des morceaux cohérents
- la recherche de passages est plus précise

#### Exemple

Un article de 2000 mots peut être divisé en 10 morceaux de 200 mots chacun.

Chaque morceau devient un élément indexé dans FAISS.

---

### 7.5 `create_vector_store(chunks, persist_directory="db/faiss_index")`

```python
def create_vector_store(chunks, persist_directory="db/faiss_index"):
```

Ici, on génère les embeddings et on les stocke.

```python
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
```

Puis :

```python
vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
```

Cela transforme chaque chunk en vecteur et l’ajoute dans FAISS.

Ensuite :

```python
os.makedirs(persist_directory, exist_ok=True)
vectorstore.save_local(persist_directory)
```

Cela enregistre la base vectorielle sur le disque pour la réutiliser plus tard.

---

### 7.6 `main()`

```python
def main():
```

Cette fonction est le cœur du pipeline.

Elle fait :

1. définir les chemins du dossier `docs` et de `db/faiss_index`
2. vérifier si la base existe déjà
3. si oui, la charger
4. sinon, lire les documents, les découper et les indexer

```python
if os.path.exists(os.path.join(persistent_directory, "index.faiss")):
```

C’est une sécurité importante : on ne recrée pas une base déjà présente.

---

## 8) Explication du code de `retrieval_pipeline.py`

### 8.1 Chargement du modèle d’embedding

```python
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
```

Cela permet d’obtenir la même représentation vectorielle que celle utilisée durant l’ingestion.

Important : le modèle utilisé pour l’indexation et pour la recherche doit être le même.

Sinon, la comparaison vectorielle ne fonctionne pas correctement.

---

### 8.2 Vérification de l’index existant

```python
if not os.path.exists(os.path.join(persistent_directory, "index.faiss")):
    raise ValueError(...)
```

Cette ligne protège le script :

- si la base n’existe pas, on ne tente pas de la charger
- on indique clairement à l’utilisateur de lancer l’ingestion d’abord

---

### 8.3 Chargement de la base FAISS

```python
db = FAISS.load_local(persistent_directory, embedding_model, allow_dangerous_deserialization=True)
```

Cette ligne lit la base vectorielle enregistrée sur le disque.

Le mot clé `allow_dangerous_deserialization=True` est spécifique à FAISS dans ce contexte, car il accepte de reconstruire les objets sauvegardés localement.

---

### 8.4 La question utilisateur

```python
query = "What was NVIDIA's first graphics accelerator called?"
```

C’est la requête test. Elle est transformée en embedding, puis comparée avec les embeddings des chunks.

---

### 8.5 Le retriever

```python
retriever = db.as_retriever(search_kwargs={"k": 5})
```

`k = 5` signifie : « retourne les 5 meilleurs passages les plus proches de la requête ».

Ensuite :

```python
relevant_docs = retriever.invoke(query)
```

Cela exécute la recherche sémantique et renvoie les meilleurs chunks.

---

### 8.6 Affichage du résultat

```python
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
```

On affiche le contenu des passages trouvés pour que l’utilisateur puisse voir le contexte.

---

## 9) Ce qu’on a appris de cette aventure

### 9.1 Les fichiers texte ne sont pas tous lus de la même manière

Le système d’exploitation et le code du fichier influencent l’encodage.

Un fichier texte peut être en UTF-8, CP1252, Latin-1, etc.

Le fait d’utiliser explicitement UTF-8 évite beaucoup de bugs.

### 9.2 Une base vectorielle est un outil de recherche, pas seulement un stockage

Le vrai rôle d’une base vectorielle est de comparer des vecteurs et de trouver les plus proches.

C’est exactement ce qui permet la recherche sémantique.

### 9.3 Les erreurs ne sont pas toujours dans le code logique

Dans ce projet, le vrai blocage initial n’était pas la logique RAG elle-même, mais :

- l’encodage des fichiers
- le backend vectoriel instable
- la mauvaise hypothèse sur l’existence de la base

Ce sont des problèmes d’environnement et d’intégration.

### 9.4 Il faut utiliser des outils compatibles avec l’environnement

On a constaté que Chroma ne fonctionnait pas correctement sur Windows dans cet environnement.

Le choix de FAISS a permis de stabiliser l’application.

---

## 10) Vision d’ensemble de la logique du projet

On peut résumer le système comme ceci :

```text
fichiers .txt -> chargement -> découpage -> embeddings -> FAISS -> recherche -> contexte -> réponse
```

### Schéma conceptuel

```text
docs/
  │
  ├─ lecture des fichiers
  │
  ├─ split en chunks
  │
  ├─ génération embeddings
  │
  ├─ sauvegarde dans db/faiss_index
  │
  └─ recherche sémantique dans la base
```

---

## 11) Ce que tu dois retenir absolument

### Points clés

- les fichiers doivent être lus en UTF-8
- la base de données vectorielle doit être vérifiée avant d’être utilisée
- les chunks doivent être cohérents et pas trop longs
- le même modèle d’embedding doit être utilisé pour l’indexation et la recherche
- FAISS est un bon choix pour un projet local et stable

### En une phrase

Le projet fonctionne parce que les documents sont bien lus, bien découpés, bien indexés et bien recherchés dans une base vectorielle adaptée à l’environnement.

---

## 12) Mini résumé pour te faire gagner du temps

Si tu veux comprendre rapidement :

- `ingestion_pipeline.py` = construit la mémoire du système
- `retrieval_pipeline.py` = demande à la mémoire de trouver les meilleurs passages
- `docs/` = base documentaire de référence
- `FAISS` = moteur de recherche vectorielle local
- `OpenAIEmbeddings` = transforme le texte en vecteurs

---

## 13) Prochaine étape recommandée

Tu peux maintenant te poser ces questions :

1. Comment faire pour indexer plus de documents ?
2. Comment ajouter une vraie interface utilisateur ?
3. Comment faire répondre l’IA avec le contexte trouvé ?
4. Comment transformer ce projet en chatbot complet ?

C’est là que commence vraiment le projet avancé : génération de réponses avec contexte.

---

## 14) Conclusion

Ce projet n’est pas seulement une suite de scripts Python. C’est une mini architecture de recherche sémantique avec :

- chargement de données
- transformation en embeddings
- stockage vectoriel
- recherche de contexte
- réponse à la question

Tu as maintenant la vision globale :

- le but du code
- les fichiers qui participent
- les erreurs rencontrées
- la logique de fonctionnement
- pourquoi le système est enfin stable

---

## 15) Références directes du projet

- [ingestion_pipeline.py](ingestion_pipeline.py)
- [retrieval_pipeline.py](retrieval_pipeline.py)
- [docs](docs)

Tu peux ouvrir les fichiers du projet pour relire chaque partie avec cette leçon en tête.
