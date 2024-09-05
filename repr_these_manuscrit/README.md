# Reproductibilité des résultats

L' intégralité des résultats présentés dans le manuscrit sont reproductibles à partir de ce dossier. Les simulations sont détaillées dans des Notebooks et sont réparties en chapitres, conformément à l'organisation du manuscrit :

 * `chapitres\chap3_caracterisation.ipynb` : résultats sur la caractérisation du réseau de nanosatellites (Chapitre 3)
 * `chapitres\chap4_division.ipynb` : résultats sur la division équitable de graphe (Chapitre 4)
 * `chapitres\chap5_reliability.ipynb` : résultats sur la fiabilité du système distribué (Chapitre 5)

Afin de reproduire les résultats du manuscrit, il suffit d'exécuter une à une les cellules des différents Notebooks.

Les scripts Python3 permettent de générer des données massives à partir des traces de nanosatellites (dossier `\data`), comme par exemple la valeur de la densité des noeuds en fonction du nombre de groupe (Chapitre 4), ou la valeur des métriques de robustesse et de résilience en fonction du temps (Chapitre 5). Il n'est pas nécessaire de les exécuter plusieurs fois : les données sont déjà exportées dans le dossier `output\data`.

Le script `chapitres\swarm_sim.py` correspond au module permettant de faire les simulations : il est importé dans les différents Notebooks.