from fastapi import APIRouter, HTTPException, status

from recsys_streaming_api.engine.utils import load_model, build_input_tensor
from recsys_streaming_api.engine.schemas import RecommendationInput, RecommendationListOutput
from recsys_streaming_api.db import client


router = APIRouter()

model = load_model(client=client)

@router.post(
    path="/recommend", 
    response_model=RecommendationListOutput,
    status_code=status.HTTP_200_OK
)
async def recommend(input_data: RecommendationInput):
    input_tensor = build_input_tensor(user_id=input_data.user_id, item_id=input_data.item_id)
    output = model(input_tensor)
    recommendation_list = output.item()
    
    return {"item_ids": [int(recommendation_list)] * 5}
