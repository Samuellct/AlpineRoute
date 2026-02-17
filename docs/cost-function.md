# Fonction de cout

## Modèle

Le coût de traversée d'un pixel est le produit de plusieurs facteurs :

$$C = f_{pente} \times f_{altitude} \times f_{aspect} \times f_{glacier} \times f_{rugosite}$$

Chaque facteur est un multiplicateur >= 1.0 (sauf pente qui est normalisée à 1.0 pr les terrain plat). Le cout final est multiplie par la distance euclidienne entre pixels voisins lors du pathfinding.

Source : `backend/alpineroute/cost/surface.py`

## Facteur pente

Le facteur principal. Basé en partie sur la hiking function de Tobler (1993).

### Vitesse de déplacement

$$v = v_0 \cdot e^{-3.5 |s + 0.05|} \cdot k$$

Ou :
- $v_0$ = 6 km/h (vitesse de base)
- $s$ = gradient
- 0.05 = gradient optimal (descente douce a ~2.86°)
- $k$ = 0.6 (facteur hors-sentier)

Le cout est normalisé pour que terrain plat = 1.0 :

$$f_{pente} = \frac{v_{plat}}{v}$$

### Pénalités supplémentaires

Au-dela de certains seuils, des multiplicateurs s'ajoutent :
- Pente > 45° : x5 (terrain très raide, progression technique)
- Pente > 60° : x50 en plus (quasi-infranchissable, *mur*)


### Reference

Tobler W. (1993). *Three presentations on geographical analysis and modeling*. Technical Report 93-1, UCSB.

## Facteur altitude

Pénalité liée à la réduction de la capacité physique en altitude.

### Formule

$$capacite = \max\left(1 - r \cdot \frac{alt - 1500}{1000},\ 0.3\right)$$

$$f_{altitude} = \frac{1}{capacite}$$

### Paramètres

- Seuil : 1500 m (en dessous, pas de penalité)
- Taux acclimaté ($r$) : 0.03 (3% de perte par 1000 m au-dessus de 1500 m)
- Taux non acclimaté ($r$) : 0.063 (6.3% par 1000 m)
- Capacite minimum : 0.3

### References

- Buskirk E.R. et al. (1966). *Maximal performance at altitude and on return from altitude in conditioned runners.* Journal of Applied Physiology.
- West J.B. (1996). *Prediction of barometric pressures at high altitude.* Journal of Applied Physiology.

## Facteur aspect / saison

Pénalité basée sur l'orientation de la pente et la période de l'année. Modalise le risque lié à l'exposition solaire.

### Eté (juin-septembre)

Les faces sud recoivent plus de soleil, ce qui augmente le risque de chutes de pierres avec le degel, de crevasses ouvertes et de neige pourrie.

$$penalite = 1 + P_{max} \cdot \max(\cos(\theta - 180), 0)$$

Ou $\theta$ est l'aspect en degrés (0 = nord). La pénalité est notée seulement si pente > 30 deg et altitude > 2500 m : $P_{max}$ = 0.5

### Hiver (octobre-mai)

Les faces nord accumulent plus de neige et recoivent moins de soleil.

$$penalite = 1 + P_{max} \cdot \max(\cos(\theta), 0)$$

Appliquée si pente > 25 deg. $P_{max}$ = 0.3

## Facteur glacier

Surcout pour les zones glaciaires, fonction de la pente locale.

### Niveaux

| Pente | Multiplicateur | Contexte |
|-------|---------------|----------|
| < 10 deg | 1.3 | Glacier plat |
| 10-20 deg | 2.0 | Pente moderée |
| 20-30 deg | 4.0 | Pente raide, risque crevasses |
| > 30 deg | 10.0 | Zone de seracs |

Le masque glacier provient du Randolph Glacier Inventory (RGI) 7.0 rasterisé sur la grille Lidar.

### Limites

Le modèle ne détecte pas les crevasses individuelles. Le surcoût est uniforme sur toute la zone glaciaire pour une tranche de pente donnée.

## Facteur rugosité

Pénalité basée sur l'irrégularité locale du terrain.

### Formule

$$f_{rugosite} = 1 + s \cdot \min(TRI, TRI_{max})$$

Ou :
- $TRI$ = Terrain Ruggedness Index, calculé comme l'écart-type des altitudes dans une fenetre 3x3
- $s$ = 0.8 (facteur d'echelle)
- $TRI_{max}$ = 5.0 m (cut pour éviter des couts infinis sur les falaises)

### Interprétation

| TRI (m) | Type de terrain | Cout |
|---------|----------------|------|
| 0 | Plat (glacier, alpage) | 1.0 |
| 1 | Irregulier léger (pierrier) | 1.8 |
| 3 | Chaotique (gros blocs, moraine) | 3.4 |
| 5+ | Max (éboulis, paroi verticale) | 5.0 |

### Reference

Riley S.J. et al. (1999). *A terrain ruggedness index that quantifies topographic heterogeneity.* Intermountain Journal of Sciences.

## Couverture du sol (WorldCover)

Multiplicateur additionnel basé sur les classes ESA WorldCover 10 m.

| Code | Classe | Multiplicateur |
|------|--------|---------------|
| 10 | Foret | 2.5 |
| 20 | Buissons | 1.8 |
| 30 | Herbe / alpage | 1.2 |
| 40 | Cultures | 1.0 |
| 50 | Bati | 20.0 |
| 60 | Moraine / eboulis | 1.5 |
| 70 | Neige / glace | 1.3 |
| 80 | Eau | 50.0 |
| 90 | Zone humide | 3.0 |
| 95 | Mangrove | 3.0 |
| 100 | Mousse / lichen | 1.3 |
| 0 | Nodata | 1.0 |

Source : `backend/alpineroute/config.py` (`WORLDCOVER_MULTIPLIERS`)

## Paramètres configurables

Tous les paramètres sont centralisés dans `backend/alpineroute/config.py` :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `TOBLER_BASE_SPEED_KMH` | 6.0 | Vitesse de base Tobler (km/h) |
| `TOBLER_OPTIMAL_GRADIENT` | 0.05 | Gradient optimal (environ 2.86° descente) |
| `OFF_TRAIL_FACTOR` | 0.6 | Reduction vitesse hors-piste |
| `STEEP_SLOPE_THRESHOLD_DEG` | 45 | Seuil pente raide (deg) |
| `STEEP_SLOPE_MULTIPLIER` | 5.0 | Multiplicateur pente raide |
| `CRITICAL_SLOPE_DEG` | 60 | Seuil pente critique (deg) |
| `CRITICAL_SLOPE_MULTIPLIER` | 50.0 | Multiplicateur pente critique |
| `HYPOXIA_ALTITUDE_THRESHOLD` | 1500 | Seuil altitude hypoxie (m) |
| `HYPOXIA_RATE_ACCLIMATIZED` | 0.03 | Taux perte acclimaté |
| `HYPOXIA_RATE_NOT_ACCLIMATIZED` | 0.063 | Taux perte non acclimaté |
| `HYPOXIA_MIN_CAPACITY` | 0.3 | Capacite O2 minimum |
| `ASPECT_SOUTH_PENALTY_MAX` | 0.5 | Penalite max face sud ete |
| `ASPECT_NORTH_PENALTY_MAX` | 0.3 | Penalite max face nord hiver |
| `GLACIER_COST_FLAT` | 1.3 | Cout glacier < 10 deg |
| `GLACIER_COST_MODERATE` | 2.0 | Cout glacier 10-20 deg |
| `GLACIER_COST_STEEP` | 4.0 | Cout glacier 20-30 deg |
| `GLACIER_COST_VERY_STEEP` | 10.0 | Cout glacier > 30 deg |
| `ROUGHNESS_SCALE` | 0.8 | Echelle cout rugosite |
| `ROUGHNESS_CLAMP` | 5.0 | TRI max (m) |

## Limites

Plusieurs points a améliorer pour les V1.1 et V2.0:

- **Modèle isotrope** : Le coût ne dépend pas de la direction de déplacement. Monter et descendre une pente à 30 degrés devrait avoir des couts différents (prévu V1.1)
- **Pas de détection de crevasses** : le masque RGI donne les contours glaciaires mais pas la structure interne. Prévu de tester une approche Deep Learning pour la V2.0 (https://doi.org/10.1016/j.jag.2025.104495).
- **Parametres WorldCover** : les multiplicateurs ne sont pas encore calibrés.
- **Saisonnalité simplifiée** : le modèle aspect/saison donne une info basique, modele python de radiation solaire en cours de tests pour la V2.0
