# LIFPROJET - SUJET AM1: Deep learning pour de la classification d'images

## FERHAOUI Yanisse 11909519 - CHOGLI Sofiane 12005328 - YOUNSI Célina 11904617

## Objectifs du projet

Ce projet est une réplique du jeu 'Quick, Draw!' de Google.

Il nous a permis de t'initier au deep learning, et plus particulièrement à s'en servir pour de la classification d'image.

## Arborescence du dépôt

```//
├── data
│   ├── draw_result.png
│   ├── model_draw.txt
│   ├── result.txt
├── static
│   ├── main.js
│   └── style.css
├── templates
│   └── index.html
├── .gitignore
├── Numbers.py
├── README.md
├── classes.txt
├── draw_images.py
├── draw_model.py
├── model_draw.h5
├── requirements.txt
├── server.py
└── test.py
```

- Le dossier 'data' regroupe quelques information sur le réseau de neurones utilisé (sa configuraton et les résultats de son entraînement).
- 'Static' et 'templates' regroupent les élements relatifs à la page web (le html, css et javascript).
- 'Numbers.py' est le "brouillon" du projet, dedans se trouve notre tout premier réseau de neurones qui permet de classifier des chiffres.
- 'classes.txt' contient tous les 345 noms de classes de dessins, utile pour les récupérer dans le serveur.
- 'draw_images.py' contient la fonction qui récupère 1200 images par classe dans la base de donnée de Google.
- 'draw_model.py' contient le réseau de neurones principal.
- 'model_draw.h5' est le fichier de du modèle entrainé dans 'draw_model.py'.
- 'server.py' est le serveur, qu'on lancera pour accéder au site.
- 'test.py' réalise la même chose que dans le serveur, mais sans l'interactivité du site web.

## Prérequis pour lancer le projet

Au préalable, il faudra installer `Python` et `Pip`.

Ensuite, ouvrez un terminal et éxécutez la commande suivante pour installer les librairies nécessaires au fonctionnement du projet:

```bash
pip install -r requirements.txt
```

## Lancement du projet

Ouvrez un terminal et situez vous à la racine du dépôt, puis éxécutez la commande suivante:

```bash
python3 server.py
```

Cela devrait lancer le serveur, lorsque le chargement est terminé, ouvrer votre navigateur web et allez à la page <http://127.0.0.1:5000/>.

## Comment utiliser le site web

Tentez de dessiner quelque chose dans la zone de dessin, lorsque ce sera terminé appuyez sur le bouton 'Prédire', un diagramme camembert apparaîtra, il correspondra à ce que le réseau de neurones a prédit.

## Liens utiles

<https://larswaechter.dev/blog/recognizing-hand-drawn-doodles/>

<https://www.tensorflow.org/tutorials/keras/classification>
