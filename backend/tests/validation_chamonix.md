# Validation qualitative -- secteur Chamonix

Checklist de validation manuelle pour verifier que les routes calculees sont coherentes avec le terrain reel. A faire avant chaque release.

## Prerequis

- App lancee localement (backend + frontend)
- Premieres dalles DEM deja en cache (sinon compter 2-5 min de telechargement)
- Fond de carte IGN topo active pour comparaison visuelle

## Route 1 : Refuge du Requin -> Aiguille du Midi

**Coordonnees** :
- Depart : 45.9068, 6.9207 (refuge du Requin, ~2516m)
- Arrivee : 45.8786, 6.8875 (Aiguille du Midi, ~3842m)

**Comportement attendu** :
- La route doit passer par le glacier du Geant (% glacier > 30%)
- Doit eviter les zones de seracs du glacier des Periades
- D+ total entre 1300 et 1600m
- Distance 2D entre 4 et 7 km
- Temps Tobler entre 3 et 6h
- La trace ne doit pas traverser de parois rocheuses verticales

**Verification visuelle** :
- Superposer la route sur la carte IGN topo
- Verifier que la trace suit un cheminement "logique" (pas de zigzags absurdes)
- La montee finale vers l'arete doit rester sur un versant praticable

## Route 2 : Traversee des Cosmiques

**Coordonnees** :
- Depart : 45.8786, 6.8875 (Aiguille du Midi, ~3842m)
- Arrivee : 45.8674, 6.8814 (Cosmiques refuge approx, ~3613m)

**Comportement attendu** :
- Trajet court (< 2 km)
- D- predominant (~200-300m)
- % glacier modere (l'arete est en grande partie rocheuse)
- Pentes traversees < 50 deg en moyenne
- Temps Tobler < 2h

**Verification visuelle** :
- La trace doit rester sur l'arete ou juste en dessous, pas descendre dans la Vallee Blanche
- Pas de passage par la face sud tres raide

## Route 3 : Plan de l'Aiguille -> Refuge du Requin (approche glacier)

**Coordonnees** :
- Depart : 45.9044, 6.8849 (Plan de l'Aiguille, ~2317m)
- Arrivee : 45.9068, 6.9207 (Refuge du Requin, ~2516m)

**Comportement attendu** :
- Traversee quasi-horizontale puis legere montee
- Distance 2-4 km
- D+ entre 200 et 500m
- Le chemin doit contourner les barres rocheuses des Perrons
- Portion glacier moderee (Mer de Glace / Envers des Aiguilles)

**Verification visuelle** :
- Le trace ne doit pas descendre sur la Mer de Glace par les echelles (trop raide)
- Doit rester en versant nord des Perrons

## Criteres de rejet

Un resultat est rejete si :
- La route passe par une pente > 60 deg sur plus de 100m
- La distance 2D est plus de 3x la distance a vol d'oiseau
- Le D+ calcule est negatif ou le D- est negatif
- Le temps Tobler est < 10 min pour un trajet de plus de 2km
- La route fait des allers-retours absurdes (visuellement)

## Procedure

1. Lancer l'app (`docker compose up` ou dev local)
2. Pour chaque route ci-dessus :
   - Placer les points depart/arrivee
   - Parametres : resolution 1m, mois juillet, acclimatise
   - Calculer et noter les stats
   - Comparer avec les valeurs attendues
   - Screenshot de la route sur fond IGN
3. Refaire avec resolution 2m pour verifier que les resultats sont qualitativement similaires
