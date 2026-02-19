# schemas Pydantic pour l'API
# source: T09 + extensions

from pydantic import BaseModel, field_validator
from typing import Optional

from alpineroute.config import (
    VALID_LAT_RANGE, VALID_LON_RANGE, VALID_RESOLUTIONS,
    MAX_ALTERNATIVE_ROUTES, ZONE_TYPES,
)


class RouteRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    resolution: float = 1.0
    month: int = 7
    acclimatized: bool = True
    n_alternatives: int = 0
    anisotropic: bool = False
    save: bool = True
    name: Optional[str] = None

    @field_validator("start_lat", "end_lat")
    @classmethod
    def check_lat(cls, v):
        lo, hi = VALID_LAT_RANGE
        if not (lo <= v <= hi):
            raise ValueError(f"latitude {v} hors range [{lo}, {hi}]")
        return v

    @field_validator("start_lon", "end_lon")
    @classmethod
    def check_lon(cls, v):
        lo, hi = VALID_LON_RANGE
        if not (lo <= v <= hi):
            raise ValueError(f"longitude {v} hors range [{lo}, {hi}]")
        return v

    @field_validator("resolution")
    @classmethod
    def check_resolution(cls, v):
        if v not in VALID_RESOLUTIONS:
            raise ValueError(f"resolution {v} invalide, accepte: {VALID_RESOLUTIONS}")
        return v

    @field_validator("month")
    @classmethod
    def check_month(cls, v):
        if not (1 <= v <= 12):
            raise ValueError(f"month {v} invalide, doit etre entre 1 et 12")
        return v

    @field_validator("n_alternatives")
    @classmethod
    def check_n_alternatives(cls, v):
        if not (0 <= v <= MAX_ALTERNATIVE_ROUTES):
            raise ValueError(f"n_alternatives doit etre entre 0 et {MAX_ALTERNATIVE_ROUTES}")
        return v


class ZoneCreate(BaseModel):
    name: str
    zone_type: str
    cost_multiplier: float = 100.0
    geojson: dict
    active: bool = True

    @field_validator("zone_type")
    @classmethod
    def check_zone_type(cls, v):
        if v not in ZONE_TYPES:
            raise ValueError(f"zone_type '{v}' invalide, accepte: {ZONE_TYPES}")
        return v


class ZoneUpdate(BaseModel):
    name: Optional[str] = None
    zone_type: Optional[str] = None
    cost_multiplier: Optional[float] = None
    geojson: Optional[dict] = None
    active: Optional[bool] = None

    @field_validator("zone_type")
    @classmethod
    def check_zone_type(cls, v):
        if v is not None and v not in ZONE_TYPES:
            raise ValueError(f"zone_type '{v}' invalide, accepte: {ZONE_TYPES}")
        return v


class HealthResponse(BaseModel):
    status: str
