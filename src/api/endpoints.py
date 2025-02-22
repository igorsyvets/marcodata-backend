from fastapi import APIRouter
from typing import Dict, List

router = APIRouter()

@router.get("/test")
async def test_endpoint() -> Dict[str, str]:
    """
    Basic test endpoint
    """
    return {"message": "API is working"}

@router.get("/items")
async def get_items() -> List[Dict[str, str]]:
    """
    Sample endpoint returning a list of items
    """
    return [
        {"id": "1", "name": "Item 1"},
        {"id": "2", "name": "Item 2"}
    ]