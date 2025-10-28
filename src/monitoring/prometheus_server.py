from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI

app = FastAPI(title="Prometheus Exporter")

instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/")
def root():
    return {"message": "📈 Metrics exporter running"}
