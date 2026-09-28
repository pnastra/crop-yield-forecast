"""FastAPI service: one-year-ahead crop yield forecast plus the last-year baseline.

Run locally with `make serve`, then open http://127.0.0.1:8000/docs.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from cropyield.predict import Forecaster

forecaster = Forecaster()  # loaded once at startup; validators below need its lists

app = FastAPI(
    title="Crop Yield Forecast",
    description="One-year-ahead national crop yield (hg/ha), with a 'same as last year' baseline.",
    version="0.1.0",
)


class PredictRequest(BaseModel):
    area: str = Field(examples=["Thailand"])
    item: str = Field(examples=["Rice, paddy"])
    year: int = Field(ge=forecaster.min_year, le=forecaster.max_year, examples=[2012])

    @field_validator("area")
    @classmethod
    def known_area(cls, v: str) -> str:
        if v not in forecaster.areas:
            raise ValueError("unknown country; GET /health lists the valid ones")
        return v

    @field_validator("item")
    @classmethod
    def known_item(cls, v: str) -> str:
        if v not in forecaster.items:
            raise ValueError(f"unknown crop; valid crops: {', '.join(forecaster.items)}")
        return v


class PredictResponse(BaseModel):
    area: str
    item: str
    year: int
    prediction_hg_ha: float
    baseline_hg_ha: float
    actual_hg_ha: float | None = Field(description="Null when the year is past the data (2014).")
    model_trained_through: int


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "years": [forecaster.min_year, forecaster.max_year],
        "crops": forecaster.items,
        "countries": forecaster.areas,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    result = forecaster.predict(req.area, req.item, req.year)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No {req.item} yield for {req.area} in {req.year - 1}, so there is nothing to forecast from.",
        )
    return result
