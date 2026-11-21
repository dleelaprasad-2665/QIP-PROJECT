# Material Intelligence

**AI-Based Visual Assessment of Construction Material Condition and Usability for Intelligent Site Resource Management**

Material Intelligence is a production-oriented Python application for academic researchers, construction engineers, quality assurance teams, and site managers. It provides dataset management, CLAHE preprocessing, YOLOv8 localization, EfficientNet-B0 condition classification, Grad-CAM explanations, training/evaluation workflows, inspection review, inventory control, and reporting — entirely in Python with a NiceGUI interface.

Legacy Severstal notebook assets in this workspace (`train.csv`, `utilities.py`, TensorFlow weights) are **not** used by Material Intelligence. The application trains and evaluates construction-specific PyTorch / Ultralytics models only.

## Supported features

- Role-based authentication (Administrator, AI Researcher, Quality Engineer, Site Supervisor, Viewer)
- Dataset Studio for Roboflow / CVAT YOLO ZIP imports, classification folders, and COCO validation
- Preprocessing Laboratory with LAB-space CLAHE and training-only augmentation
- Asynchronous model training jobs (YOLOv8 + EfficientNet-B0)
- Evaluation dashboards using real scikit-learn / Ultralytics metrics (no fabricated scores)
- Material inspection pipeline with auditable engineering review statuses
- Inventory batches with quarantine / approve / reject stock movements
- CSV and PDF reporting from persisted records
- Grad-CAM visualizations via Captum

## Architecture

```
material_intelligence/
  app/
    main.py              # NiceGUI entrypoint
    config.py            # Pydantic settings
    ui/                  # NiceGUI pages, theme, components
    api/                 # Health/readiness routes
    domain/              # Enums and schemas
    services/            # Business workflows
    ml/                  # Preprocessing, YOLO, EfficientNet, Grad-CAM, eval
    db/                  # SQLAlchemy models and session
    jobs/                # Background training worker
    security/            # AuthN / AuthZ
  tests/
migrations/              # Alembic
```

Boundaries: UI → services → DB / ML. Training never runs inside NiceGUI request callbacks; jobs are queued to a background worker.

## Technology stack

Python 3.11+, NiceGUI, FastAPI (health routes), PyTorch / Torchvision, Ultralytics YOLOv8, OpenCV, Albumentations, Captum, scikit-learn, Plotly, SQLAlchemy, Alembic, Pydantic, Pillow, NumPy, Pandas, ReportLab, Pytest, Ruff.

## Prerequisites

- Python 3.11 or 3.12
- CPU is sufficient for development (YOLOv8n + small EfficientNet runs)
- Optional CUDA GPU for faster training
- Annotated construction-material datasets exported from Roboflow or CVAT

## Setup

```bash
cd QIP
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate

python -m pip install --upgrade pip

# Recommended on CPU-only / disk-constrained machines:
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir -e ".[dev]"

copy .env.example .env   # Windows
# cp .env.example .env   # Linux / macOS
```

Edit `.env` and change `SECRET_KEY` and `BOOTSTRAP_ADMIN_PASSWORD` before any shared deployment.

## Database initialization

SQLite (default local development):

```bash
python -c "from material_intelligence.app.services.bootstrap import bootstrap_application; bootstrap_application()"
```

The application also bootstraps automatically on startup (admin user, default material/defect taxonomy, demo site `DEMO-01`).

PostgreSQL (production):

1. Set `DATABASE_URL=postgresql+psycopg://user:password@host:5432/material_intelligence`
2. Install a PostgreSQL driver (`pip install psycopg[binary]`)
3. Run `alembic revision --autogenerate -m "initial"` then `alembic upgrade head`

## Optional demo data (explicitly labeled)

For dashboard development only. Demo records use site code `DEMO-DATA` and are labeled **DEMO DATA — not operational**.

```bash
material-intelligence-seed-demo
# or force recreate:
python -m material_intelligence.app.services.demo_seed --force
```

Toggle **Include demo data** on the Overview dashboard to hide them from KPIs and tables.

## Start the application

```bash
python -m material_intelligence.app.main
```

Or:

```bash
material-intelligence
```

Open `http://127.0.0.1:8080` and sign in with the bootstrap administrator credentials from `.env`.

Health checks:

- `GET /api/health`
- `GET /api/ready`

## Dataset import workflow

1. Annotate images in **Roboflow** or **CVAT** (external tools; no API account is connected by default).
2. Export YOLO detection format (images + labels + `data.yaml`) or build classification folders named by condition.
3. Open **Dataset Studio** and upload the ZIP or class images.
4. Review validation: missing labels, corrupt images, duplicates, class imbalance.
5. Only datasets marked ready should be used for training.

Initial material / defect taxonomy:

| Material | Conditions |
|----------|------------|
| Clay brick | cracked, chipped, broken, no_visible_damage |
| Steel rebar | surface_corrosion, extensive_visible_corrosion, deformation, no_visible_damage |
| Cement bag | torn_packaging, visible_moisture_staining, damaged_packaging, no_visible_damage |

Administrators can add materials and defect classes in **Administration** without code changes.

## Model training

1. Import a validated dataset.
2. Open **Model Training**.
3. Choose EfficientNet-B0 (classification folders) or YOLOv8 (YOLO `data.yaml`).
4. Configure epochs, batch size, learning rate, image size, device.
5. Start the job — status progresses Queued → Running → Completed / Failed.
6. Select a completed checkpoint and **Set active for inference**.

CPU-friendly defaults: YOLOv8n, small batch sizes, few epochs for smoke experiments.

**Important:** ImageNet / COCO pretrained weights are initialization only. Until a task-specific checkpoint is trained and registered, the UI shows **Model Not Trained**.

## Model evaluation

1. Open **Model Evaluation**.
2. Select a ready checkpoint and a held-out / validation dataset.
3. Run evaluation.
4. Inspect confusion matrices, per-class F1, mAP (YOLO), and stored records.

Dashboards display **Not evaluated** until a real evaluation completes. No metric values are fabricated.

## Inspection workflow

1. Select a construction site and optional material batch.
2. Upload a material image in **Material Inspection**.
3. YOLOv8 localizes regions (when a detector is active).
4. EfficientNet-B0 classifies supported crops.
5. Optional Grad-CAM is generated on request.
6. Recommended operational status is stored (screening completed, human inspection required, additional testing required, etc.).
7. Authorized engineers finalize decisions in **Quality Review** (audited).
8. Approved / quarantined / rejected quantities update inventory only through that workflow.

Visual AI screening never certifies structural safety.

## Reports

Generate inspections CSV, inventory CSV, and site PDF reports from persisted records in **Reports**.

## Testing

```bash
pytest
ruff check material_intelligence
```

Tests use tiny tensors/datasets and do not download multi-gigabyte models or run expensive training jobs. Grad-CAM and EfficientNet tests train a one-epoch miniature model.

## Security configuration

- Store secrets only in environment variables / `.env` (never commit real credentials)
- Change bootstrap admin password immediately on shared systems
- Upload validation enforces type, size, image decode checks, and ZIP path-traversal protection
- Model loading uses trusted artifact paths from the registry — do not load arbitrary user-provided pickled models
- Role checks protect training, dataset changes, approvals, and administration

## Troubleshooting

| Issue | Guidance |
|-------|----------|
| Login fails | Confirm bootstrap credentials in `.env` and that bootstrap ran |
| Model Not Trained | Train and activate a checkpoint; COCO/ImageNet weights are not treated as ready material models |
| YOLO training fails | Ensure dataset contains `data.yaml` / `dataset.yaml` with train/val paths |
| EfficientNet training fails | Use classification folder structure (`class_name/image.jpg`) with ≥2 classes |
| CUDA errors | Set device to `cpu` in training config |
| Grad-CAM unavailable | Requires a ready EfficientNet-B0 checkpoint |
| Port in use | Change `APP_PORT` in `.env` |

## Known limitations

- Roboflow/CVAT are integrated through export formats only (no live API sync yet)
- Local training worker is an in-process daemon thread suitable for single-host development, not a distributed queue
- CPU training is intentionally slow; use small datasets for demos
- Detection→classification material alignment requires consistent class naming / `material_key` configuration
- Legacy TensorFlow Severstal assets in the repository root are unrelated to this application

## Obtaining annotated datasets

1. Capture site photos of clay bricks, steel reinforcement, and cement bags under realistic lighting.
2. Create projects in Roboflow or CVAT with the defect classes above.
3. Annotate bounding boxes (detection) and/or cropped condition labels (classification).
4. Export YOLO ZIP / folder datasets and import them in Dataset Studio.
5. Split by site / batch / capture session when possible to reduce leakage before evaluation.
