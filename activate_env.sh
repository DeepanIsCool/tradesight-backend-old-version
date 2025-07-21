#!/bin/bash

# TradeSight Backend Virtual Environment Activation Script
echo "🚀 Activating TradeSight Backend Virtual Environment..."

# Activate the virtual environment
source .venv/bin/activate

echo "✅ Virtual environment activated!"
echo "📍 Python version: $(python --version)"
echo "📦 Virtual environment path: $(which python)"

echo ""
echo "🔧 Available commands:"
echo "  • Start development server: uv run uvicorn app.main:app --reload"
echo "  • Run with custom host/port: uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo "  • View API docs: http://localhost:8000/docs"
echo "  • Alternative docs: http://localhost:8000/redoc"
echo ""
echo "📚 Project uses Python 3.12 with the following key packages:"
echo "  • FastAPI $(python -c 'import fastapi; print(fastapi.__version__)')"
echo "  • TensorFlow $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not available')"
echo "  • Pandas $(python -c 'import pandas as pd; print(pd.__version__)')"
echo "  • SQLAlchemy $(python -c 'import sqlalchemy; print(sqlalchemy.__version__)')"
echo ""
echo "🎯 To deactivate: type 'deactivate'"
