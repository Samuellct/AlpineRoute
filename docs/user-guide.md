# Guide utilisateur

Guide pour utiliser AlpineRoute.

## Calcul d'un itinéraire

1. Ouvrir l'application dans le navigateur (`http://localhost:5173` en dev local, ou `http://localhost:3000` via Docker).
2. Placer le départ et l'arrivée : cliquer sur la carte (marqueur vert = départ, rouge = arrivée, ou taper un nom de lieu dans les champs texte (recherche Nominatim)).
3. Dans le panneau latéral (onglet "Calcul"), ajuster les paramètres si besoin (voir section *Paramètres*).
4. Cliquer sur **Calculer**. Une barre de progression affiche l'avancement du pipeline (Téléchargement Lidar, analyse terrain, calcul de coût, pathfinding...).
5. Le calcul prend en moyenne entre 10s et 3min selon la taille de la zone et la résolution choisie. Le premier calcul sur une zone est plus lent car il faut télécharger les dalles lidar depuis l'API IGN.

## Résultats

Une fois le calcul terminé :

- La route optimale s'affiche sur la carte.
- Si des routes alternatives ont été demandées, elles s'affichent en gris. Cliquer sur une route alternative pour la sélectionner.
- Un **badge stratégie** indique le mode de routage utilisé : Réseau (vert, tout Valhalla), Hybride (bleu, Valhalla + raster), Raster (orange, pathfinding complet sur grille lidar).
- Le panneau latéral affiche les **stats** : distance, dénivelé positif/négatif, temps estimé, pourcentage de glacier, coût total, temps de calcul.
- Le **profil altimétrique** apparaît en bas de la carte. Survoler le profil pour voir la position correspondante sur la carte.

## Export

Deux boutons d'export sont disponibles sous les stats :

- **GeoJSON** : télécharge le trace au format GeoJSON (/!\ Utilisé pendant la beta, prevu de le delete à la V1.1 ou de laisser l'option pr le mode debug).
- **GPX** : télécharge le trace au format GPX, compatible avec les montres GPS notamment.

## Fonds de carte

Le selecteur en bas a droite de la carte permet de choisir entre 4 fonds :

- **Plan IGN** : carte topographique IGN (sentiers, courbes de niveau).
- **Satellite IGN** : imagerie aérienne IGN.
- **Pentes** : couche de pentes coloree (Doublon aavec le calque pentes, prévu de le supprimer en V1.1).
- **Topo**, **Satellite** : fonds de carte MapTiler utilisée pour l'étranger.

## Zones de danger

Les zones de danger permettent de définir des polygones où le cout de passage est augmenté, ce qui force le pathfinding à les éviter.

1. Dans le panneau lateral, section "Zones de danger", cliquer sur **Ajouter une zone**.
2. Dessiner un polygone sur la carte et double-cliquer pour fermer le polygone.
3. Renseigner le nom, le type (crevasse, serac, corniche, chute de pierres, interdit, custom) et le multiplicateur de cout.
4. Un multiplicateur de 100 rend la zone très couteuse (la route la contournera). Un multiplicateur de 1000 la rend quasi-infranchissable.
5. Les zones actives sont appliquées à chaque calcul.

Tags de zone :
- `crevasse`
- `serac`
- `cornice`
- `rockfall`
- `Interdit` : zone interdite (multiplicateur eleve par defaut)
- `custom` : autre danger defini par l'utilisateur

## Historique

L'onglet "Historique" dans le panneau lateral liste les routes calculées précédemment (stockées en SQLite locale).

- Cliquer sur une route pour la recharger et l'afficher sur la carte.
- Le bouton de suppression efface définitivement une route de l'historique.
- Les routes sont triées par date (plus récentes en premier).

## Paramètres et leur effet

| Paramètre | Valeurs | Effet |
|-----------|---------|-------|
| Resolution MNT | 0.5, 1, 2, 5, 10 m | Plus la résolution est fine, plus le calcul est precis mais lent. 1m est un bon compromis. 0.5m = natif Lidar HD. |
| Mois | Janvier - Decembre | Influence le facteur d'orientation : en été (juin-sept), les faces sud en altitude sont pénalisées (risque fonte/serac). En hiver, les faces nord raides sont pénalisées (glace, mauvaises conditions). |
| Routes alternatives | 0-5 | Nombre de traces supplémentaires calculées via la méthode de pénalité. |
| Acclimatation altitude | oui/non | Modifie le facteur hypoxie au-dessus de 1500m. Une non-aclimatation entraîne une pénalité plus forte en altitude. |
| Terrain 3D | on/off | Active la vue en relief sur la carte. |

## Calques

Les calques sont accessibles depuis le sélecteur en bas à droite de la carte :

- **Pentes** : couche de pentes coloree.
- **Glaciers** : contours glaciaires RGI 7.0 en bleu semi-transparent. Se rafraîchit au déplacement de la carte.
- **Cout** : heatmap de la surface de cout du chemin (vert = facile, rouge = difficile). Disponible uniquement après un calcul de trace.
- **Traces alpinisme** : courses indexées depuis les fichiers GPX, colorées par cotation. Tooltip au survol.
- **Segments terrain** : connexions locales (échelles, passages glaciers, etc.) en pointillé jaune.

## Limites connues

- **Dépendance Valhalla** : le routage réseau et hybride nécessite Valhalla (lancé par Docker). Sans Valhalla, le pipeline bascule automatiquement en raster pur.
- **Zone de calcul** : La zone s'étend à 4km autour des points de départ et d'arrivée.
- **Bbox max 30km** : La zone de calcul est limitée à 30x30 km. Pour des traversées plus longues, découper en étapes.
- **Premier calcul lent** : Le téléchargement des dalles Lidar peut prendre 1-5 min au premier usage sur une zone. Les dalles sont ensuite en cache local.
- **WorldCover 10m** : la couche land cover (foret, moraine, eau...) a une résolution de 10m, plus grossière que le fond de carte.
- **Précision glaciers** : les contours glaciaires viennent du RGI 7.0 (données jusqu'en 2021). Les glaciers évoluent vite, les contours peuvent être en retard de quelques années.
