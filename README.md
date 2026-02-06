# Resonate Microservice

AI-powered diagnostics parser and fitness/nutrition generator service.

## Features

- 🩸 **Blood Report Parser** - Extract biomarkers from PDF reports using GPT-4 Vision
- 🏋️ **Workout Generator** - AI-personalized workout plans
- 🥗 **Nutrition Planner** - Daily meal plans based on user profile
- 📸 **Food Analyzer** - Nutritional analysis from food images

## Tech Stack

- **Framework**: FastAPI
- **AI**: OpenAI GPT-4.1-mini
- **PDF Processing**: PyMuPDF (fitz)
- **Image Processing**: Pillow

## Project Structure

```
Resonate-Microservice/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── core/
│   │   ├── config.py        # Configuration & constants
│   │   └── logger.py        # Structured logging
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── pdf_service.py   # PDF download & image conversion
│   │   └── openai_service.py # OpenAI API interactions
│   └── routes/
│       ├── parser.py        # /parse-report endpoint
│       ├── workout.py       # /generate-workout endpoint
│       └── nutrition.py     # /generate-nutrition, /analyze-food endpoints
├── tests/                   # Test files (TODO)
├── requirements.txt
├── .env
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

1. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PORT=10000
   ```

4. Run the server:
   ```bash
   uvicorn app.main:app --reload --port 10000
   ```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Detailed health status |
| `POST` | `/parse-report` | Parse blood report PDF |
| `POST` | `/generate-workout` | Generate AI workout plan |
| `POST` | `/generate-nutrition` | Generate daily meal plan |
| `POST` | `/analyze-food` | Analyze food image for nutrition |

## API Examples

### Parse Blood Report
```bash
curl -X POST http://localhost:10000/parse-report \
  -H "Content-Type: application/json" \
  -d '{"pdfUrl": "https://example.com/report.pdf", "biomarkers": ["Vitamin D", "Iron"]}'
```

### Generate Workout
```bash
curl -X POST http://localhost:10000/generate-workout \
  -H "Content-Type: application/json" \
  -d '{"fitnessLevel": "intermediate", "equipment": ["dumbbells"], "timeAvailable": 30, "injuries": []}'
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4 Vision |
| `PORT` | No | Server port (default: 10000) |

## License

Proprietary - Resonate Health
