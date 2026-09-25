# Zepto Data & AI Platform

## Project Overview

This project implements a complete data and AI workflow consisting of data collection, data cleaning, database storage, analytics, machine learning, and a GenAI-based support assistant.

## Project Modules

### Module 1 - Data Pipeline

This module collects book data using Requests and BeautifulSoup, cleans and transforms the data, converts GBP prices to INR using the fixed conversion rate, stores the data in SQLite, and performs SQL analysis.

Main components:
- Web scraping
- Data cleaning
- GBP to INR conversion
- SQLite database
- SQL queries
- Pandas analysis

### Module 2 - Analytics

This module performs exploratory data analysis and machine learning using the Titanic dataset.

Main components:
- Exploratory Data Analysis
- Missing-value handling
- Outlier analysis
- Correlation analysis
- Data visualization
- Logistic Regression
- Decision Tree
- Random Forest
- Model evaluation
- Regression
- Model persistence

### Module 3 - Support Assistant

This module implements a Zepto policy support assistant using local embeddings, ChromaDB, LangGraph, Pydantic and FastAPI.

Main components:
- Policy document ingestion
- Text chunking
- Sentence Transformer embeddings
- ChromaDB retrieval
- Intent classification
- LangGraph workflow
- FastAPI API
- Mock LLM mode

## Project Structure

```text
zepto-ai-ml-capstone/
│
├── data_pipeline/
│
├── analytics/
│
└── support_assistant/
