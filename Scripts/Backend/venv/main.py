from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI(
    title="Sistema de Detección de Cambios Geoespaciales",
    description="API para ejecutar el modelo de CNN y gestionar alertas del Parque Nacional Tunari."
)


origins = [
    "http://localhost:5173", 
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeoAlert(BaseModel):
    id: int
    latitude: float
    longitude: float
    change_type: str 
    date_detected: str
    confidence_score: float  
    
class AlertList(BaseModel):
    alerts: List[GeoAlert]



@app.get("/")
def read_root():
    return {"message": "FastAPI Backend funcionando. ¡Conectado al monitoreo del PNT!"}

@app.get("/alerts/latest", response_model=AlertList)
def get_latest_alerts():
    # SIMULACIÓN DE DATOS 
    simulated_alerts = [
        GeoAlert(
            id=101,
            latitude=-17.3888,
            longitude=-66.1950,
            change_type="Loteamiento",
            date_detected="2025-11-28",
            confidence_score=0.92
        ),
        GeoAlert(
            id=102,
            latitude=-17.3910,
            longitude=-66.1985,
            change_type="Deforestación",
            date_detected="2025-11-27",
            confidence_score=0.85
        )
    ]
    
    return {"alerts": simulated_alerts}

if __name__ == "__main__":
    # http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)