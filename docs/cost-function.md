# Fonction de cout

## Modèle

Le coût de traversée d'un pixel est le produit de plusieurs facteurs :

$$C = f_{pente} \times f_{altitude} \times f_{radiation} \times f_{glacier} \times f_{rugosité} \times f_{devers}$$

Chaque facteur est un multiplicateur >= 1.0 (sauf pente qui est normalisée à 1.0 pr les terrain plat). Le cout final est multiplie par la distance euclidienne entre pixels voisins lors du pathfinding.

Source : `backend/alpineroute/cost/surface.py`

## Facteur pente

Le facteur principal. Basé en partie sur la hiking function de Tobler (1993).

### Vitesse de déplacement

$$v = v_0 \cdot e^{-3.5 |s + 0.05|}$$

Ou :
- $v_0$ = 6 km/h (vitesse de base)
- $s$ = gradient
- 0.05 = gradient optimal (descente douce a ~2.86°)

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

## Facteur radiation solaire

Remplace l'ancien facteur aspect/saison (cosinus simple) par un modele physique avec ombres portees. Quand le calcul de radiation echoue (DEM trop petit, etc.), l'ancien modele cosinus est utilise en fallback.

### Principe

1. **Position solaire** (algo NOAA simplifie) : calcule l'elevation et l'azimut du soleil toutes les 30 min sur une journee representative (le 15 du mois).
2. **Angles d'horizon** : pour chaque pixel, balayage radial vectorise dans 36 directions. On mesure l'angle d'elevation maximum vers les reliefs environnants (portee ~5 km). Calcule sur un DEM sous-echantillonne a 5 m pour les performances.
3. **Ombres portees** : un pixel est a l'ombre si l'elevation du soleil est inferieure a l'angle d'horizon dans la direction du soleil (interpolation entre les 2 azimuths encadrants).
4. **Irradiance directe** : loi du cosinus de l'angle d'incidence sur surface inclinee, mise a zero si a l'ombre.
5. **Integration journaliere** : cumul de l'irradiance sur la journee (pas de 30 min, 4h-22h UTC).

### Cout

La radiation journaliere normalisee est transformee en facteur multiplicatif :

- **Ete** (juin-sept) : les faces tres exposees sont penalisees (neige molle, degel, chutes de pierres). Penalite max +50%.
- **Hiver** (oct-mai) : les faces ombrees sont penalisees (verglas, neige dure). Penalite max +30%.

Conditions : pente > 15 deg **et** altitude > 2000 m. En dessous, pas de penalite (1.0).

### Cache

- **Horizons** : cache permanent par zone/resolution (ne depend pas du mois)
- **Radiation mensuelle** : cache par zone/resolution/mois

### Fallback (ancien modele aspect/saison)

Si la radiation n'est pas calculable, le facteur aspect cosinus est utilise :

- Ete : penalite faces sud, $P_{max}$ = 0.5 (pente > 30 deg, alt > 2500 m)
- Hiver : penalite faces nord, $P_{max}$ = 0.3 (pente > 25 deg)

## Facteur devers (hill slope)

Penalite progressive pour les pentes laterales (traversees en devers). Onset a 25 deg.

$$f_{devers} = \begin{cases} 1.0 & \text{si } pente \leq 25° \\ 1 + 0.8 \cdot \frac{pente - 25}{30} & \text{si } pente > 25° \end{cases}$$

Le devers est penalise car la progression laterale sur terrain raide est plus lente et plus dangereuse que la montee/descente directe (qui est deja capturee par Tobler).

## Facteur glacier

Surcout pour les zones glaciaires, fonction de la pente locale.

### Niveaux

| Pente | Multiplicateur | Contexte |
|-------|---------------|----------|
| < 10 deg | 3.0 | Glacier plat (crevasses, encordement) |
| 10-20 deg | 5.0 | Pente moderee |
| 20-30 deg | 10.0 | Pente raide, zone de seracs |
| > 30 deg | 25.0 | Chutes de seracs, quasi-infranchissable |

Le masque glacier provient du Randolph Glacier Inventory 7 rasterisé sur la grille Lidar.

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
| 10 | Foret | 3.5 |
| 20 | Buissons | 2.5 |
| 30 | Herbe / alpage | 1.1 |
| 40 | Cultures | 1.5 |
| 50 | Bati | 1e6 (infranchissable) |
| 60 | Moraine / eboulis | 3.0 |
| 70 | Neige / glace | 1.5 |
| 80 | Eau | 1e6 (infranchissable) |
| 90 | Zone humide | 5.0 |
| 95 | Mangrove | 3.0 |
| 100 | Mousse / lichen | 1.2 |
| 0 | Nodata | 1.0 |

Source : `backend/alpineroute/config.py` (`WORLDCOVER_MULTIPLIERS`)

## Paramètres configurables

Tous les paramètres sont centralisés dans `backend/alpineroute/config.py` :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `TOBLER_BASE_SPEED_KMH` | 6.0 | Vitesse de base Tobler (km/h) |
| `TOBLER_OPTIMAL_GRADIENT` | 0.05 | Gradient optimal (environ 2.86° descente) |
| `STEEP_ONSET_DEG` | 35 | Debut penalite progressive (deg) |
| `STEEP_FULL_DEG` | 55 | Penalite maximale atteinte (deg) |
| `STEEP_MAX_MULTIPLIER` | 20.0 | Multiplicateur max pente raide |
| `HYPOXIA_ALTITUDE_THRESHOLD` | 1500 | Seuil altitude hypoxie (m) |
| `HYPOXIA_RATE_ACCLIMATIZED` | 0.03 | Taux perte acclimaté |
| `HYPOXIA_RATE_NOT_ACCLIMATIZED` | 0.063 | Taux perte non acclimaté |
| `HYPOXIA_MIN_CAPACITY` | 0.3 | Capacite O2 minimum |
| `RADIATION_SUMMER_PENALTY` | 0.5 | Penalite max radiation ete |
| `RADIATION_WINTER_PENALTY` | 0.3 | Penalite max radiation hiver |
| `RADIATION_SLOPE_THRESHOLD` | 15 | Seuil pente radiation (deg) |
| `RADIATION_ALTITUDE_THRESHOLD` | 2000 | Seuil altitude radiation (m) |
| `RADIATION_N_AZIMUTHS` | 36 | Nb directions balayage horizon |
| `HILLSLOPE_ONSET_DEG` | 25 | Debut penalite devers (deg) |
| `HILLSLOPE_SCALE` | 0.8 | Raideur penalite devers |
| `GLACIER_COST_FLAT` | 3.0 | Cout glacier < 10 deg |
| `GLACIER_COST_MODERATE` | 5.0 | Cout glacier 10-20 deg |
| `GLACIER_COST_STEEP` | 10.0 | Cout glacier 20-30 deg |
| `GLACIER_COST_VERY_STEEP` | 25.0 | Cout glacier > 30 deg |
| `ROUGHNESS_SCALE` | 0.8 | Echelle cout rugosite |
| `ROUGHNESS_CLAMP` | 5.0 | TRI max (m) |

## Limites

Plusieurs points a améliorer pour les V1.1 et V2.0:

- **Pas de détection de crevasses** : le masque RGI donne les contours glaciaires mais pas la structure interne. Prévu de tester une approche Deep Learning pour la V2.0 (https://doi.org/10.1016/j.jag.2025.104495).
- **Radiation simplifiée** : le modele calcule l'irradiance directe mais pas la diffuse (ciel couvert, reflexions). Suffisant pour discriminer faces exposees/ombrees en haute montagne.
