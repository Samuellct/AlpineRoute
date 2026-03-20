# Traces GPX alpinisme

Système d'indexation de traces GPX pour le graphe overlay et l'affichage sur la carte. Les traces servent à la fois de couche visuelle et de réseau topologique pour le routage hybride.

## Structure

```
data/gpx/
  index.json            # index de toutes les traces
  mont-blanc/           # dossiers par massif
    cosmiques.gpx
    ...
  grand-paradis/
    ...
  segments/             # segments terrain
    montenvers_mer_de_glace.gpx
    ...
```

## Format index.json

L'index est une liste d'entrées JSON. Deux types :

### Route (course complète)

```json
{
    "type": "route",
    "gpx_file": "mont-blanc/cosmiques.gpx",
    "massif": "Mont-Blanc",
    "summit": "Aiguille du Midi",
    "voie": "Arête des Cosmiques",
    "grade": "AD",
    "notes": "optionnel"
}
```

### Segment (connexion locale)

```json
{
    "type": "segment",
    "gpx_file": "segments/montenvers_mer_de_glace.gpx",
    "start_name": "Montenvers (gare)",
    "end_name": "Mer de Glace (pied des echelles)",
    "segment_type": "via_ferrata",
    "trail_cost": 0.40,
    "notes": "optionnel"
}
```

Le `trail_cost` est le multiplicateur appliqué sur la surface de cout (meme échelle que les sentiers OSM t1-t6).

## Ajouter une trace

1. Placer le fichier `.gpx` dans le bon sous-dossier de `data/gpx/`
2. Ajouter l'entrée dans `index.json`
3. Recharger l'index : `POST /admin/reload-index`

Le reload recalcule les stats (distance, D+), extrait le GeoJSON et synchronise la base SQLite. Le graph GPX est reconstruit.

## Cotations supportées

Echelle IFAS (International French Adjectival System) :

F, F+, PD-, PD, PD+, AD-, AD, AD+, D-, D, D+, TD-, TD, TD+, ED-, ED, ED+, ABO-, ABO, ABO+
