#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Check command
case "$1" in
    train)
        echo "🚀 Starting model training (sarcasm + category + sentiment)..."
        echo "✨ Using sarcasm-aware sentiment prediction!"
        python train.py
        ;;
    test)
        echo "🧪 Testing model on 500 test samples..."
        python test.py
        ;;
    ui)
        echo "🎨 Starting Streamlit UI..."
        streamlit run app/ui.py
        ;;
    api)
        echo "🚀 Starting FastAPI server..."
        uvicorn app.api:app --reload
        ;;
    predict)
        echo "🔮 Running inference example..."
        python inference.py
        ;;
    install)
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        ;;
    *)
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  train    - Train multi-task model (sarcasm + category + sentiment)"
        echo "  test     - Test model on 500 samples with detailed metrics"
        echo "  ui       - Run Streamlit web interface"
        echo "  api      - Run FastAPI REST API server"
        echo "  predict  - Run inference example from command line"
        echo "  install  - Install all dependencies"
        exit 1
        ;;
esac
