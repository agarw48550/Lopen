from fastapi import FastAPI
from fastapi.routing import APIRouter

app = FastAPI()
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)