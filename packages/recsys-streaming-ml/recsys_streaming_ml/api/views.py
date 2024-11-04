import subprocess
import os

from fastapi import APIRouter, HTTPException, status

from recsys_streaming_ml.db import mongo_db, read_latest_model_version_document

router = APIRouter()

@router.post(
    path="/retrain", 
    status_code=status.HTTP_200_OK
)
async def retrain(epochs: int):
    try:
        retrain_command = f"python -m run --script train -e {epochs}"
        if os.getenv("IS_CONTAINER") is None:
            shell_init = "poetry shell && "
            retrain_command = shell_init + retrain_command

        subprocess.run(retrain_command, shell=True)
        
        return {"status": "success"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    path="/version", 
    status_code=status.HTTP_200_OK
)
async def current_model_version():
    latest_model = read_latest_model_version_document(mongo_db)
    return {
        "model": latest_model['model'],
        "timestamp": latest_model["timestamp"],
        "version": latest_model["version"]
    }