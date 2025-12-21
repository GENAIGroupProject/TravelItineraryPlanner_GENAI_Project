# Travel Itinerary Project

## Overview
The **Travel Itinerary Project** is a Python-based application designed to generate personalized travel itineraries using Large Language Models (LLMs). The project leverages **Ollama** to run LLMs locally, ensuring privacy, flexibility, and offline capabilities.

This project demonstrates how to:
- Set up a Python virtual environment
- Integrate Ollama with Python
- Structure a simple AI-powered application

---

## Prerequisites
Before starting, ensure you have the following installed:

- **Python 3.9+**
- **Git**
- **Ollama** (for running LLMs locally)

---

## Step 1: Install Ollama

### On Linux / macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### On Windows
Download and install Ollama from:
```
https://ollama.com/download
```

### Verify Installation
```bash
ollama --version
```

### Pull a Model (Example)
```bash
ollama pull llama3
```

---

## Step 2: Create a Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Linux / macOS
```bash
source venv/bin/activate
```

### Windows
```bash
venv\Scripts\activate
```

---

## Step 3: Run `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 4: Run the Application

Ensure Ollama is running in the background:

```bash
ollama serve
```

Then execute the project:

```bash
python main.py
```

---

## Example Output

```
Day 1: Explore historic landmarks and local cuisine...
Day 2: Visit museums and cultural districts...
Day 3: Relax, shopping, and food experiences...
```


