from fastapi import FastAPI

from recsys_streaming_ml.api import recommendation_router


app = FastAPI() 
app.include_router(recommendation_router, prefix="/engine")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)