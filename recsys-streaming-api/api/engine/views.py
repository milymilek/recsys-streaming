from fastapi import APIRouter, HTTPException, status

import torch

from api.engine.utils import load_model, build_input_tensor
from api.engine.schemas import RecommendationInput, RecommendationListOutput
from api.config import MODEL_URI_PATH

router = APIRouter()

model = load_model(MODEL_URI_PATH)

print(model)

@router.post(
    path="/recommend", 
    response_model=RecommendationListOutput,
    status_code=status.HTTP_200_OK
)
async def recommend(input_data: RecommendationInput):
    input_tensor = build_input_tensor(user_id=input_data.user_id, item_id=input_data.item_id)

    print(input_tensor)

    output = model(input_tensor)

    recommendation_list = output.item()
    print(recommendation_list)

    return {"item_ids": [int(recommendation_list)] * 5}
