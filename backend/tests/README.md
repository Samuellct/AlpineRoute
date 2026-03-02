# Tests

## Lancer les tests

```bash
cd backend
conda run -n alpineroute pytest tests/ -v
```

## Structure

| Fichier | Contenu |
|---------|---------|
| `conftest.py` | Fixtures partagées (DEM synthétiques, transform, DB temp) |
| `test_cost.py` | Surface de cout : Tobler, hypoxie, aspect, glacier, rugosité |
| `test_utils.py` | Fonctions utilitaires |
| `test_terrain.py` | Analyse terrain (slope/aspect Horn) |
| `test_pathfinding.py` | Dijkstra isotrope + anisotrope |
| `test_pipeline.py` | Intégration pipeline (mocks réseau, mini-DEM) |
| `test_api.py` | Endpoints FastAPI |
| `test_db.py` | SQLite CRUD routes + zones |

## Fixtures (conftest.py)

- `tiny_dem` : plan incliné 3000-3500m 20x20 avec bruit
- `flat_dem` : terrain plat 3000m 20x20
- `fake_transform` : Affiné 1m centré sur Chamonix
- `glacier_mask` : bande True au centre
- `tmp_db` : SQLite temporaire avec tables initialisées
