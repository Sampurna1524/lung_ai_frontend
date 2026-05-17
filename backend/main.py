from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.inference import router
from models.loader import load_models

# 🔥 create app with metadata
app = FastAPI(
    title="Lung AI Backend",
    description="AI-powered lung cancer detection system",
    version="1.0.0"
)

# 🔥 CORS (important if frontend is hosted separately later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 load models on startup
@app.on_event("startup")
def startup_event():
    try:
        print("🚀 Starting backend...")
        load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print("❌ Error loading models:", str(e))

# 🔥 include API routes
app.include_router(router, prefix="/api")

# 🔥 root endpoint
@app.get("/")
def home():
    return {"message": "Backend running"}

# 🔥 health check (VERY useful)
@app.get("/health")
def health_check():
    return {"status": "ok"}