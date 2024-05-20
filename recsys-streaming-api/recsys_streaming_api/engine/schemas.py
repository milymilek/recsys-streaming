from pydantic import BaseModel


class RecommendationInput(BaseModel):
    user_id: int
    item_id: int


class RecommendationListOutput(BaseModel):
    item_ids: list[int]