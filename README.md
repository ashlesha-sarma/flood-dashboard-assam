# FloodSense Assam  
District-Level Flood Risk Intelligence Dashboard

Real-time ML-powered flood risk prediction across all 33 districts of Assam, monitoring 8 major rivers with 3-day forecasts and crop-impact estimates.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-floodsense--assam.onrender.com-0062FF?style=for-the-badge&logo=render&logoColor=white)](https://flood-dashboard-assam.onrender.com)

---

### 🧱 Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Backend-Flask-000000?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-316192?style=for-the-badge&logo=postgresql)
![Leaflet](https://img.shields.io/badge/Maps-Leaflet-199900?style=for-the-badge&logo=leaflet)
![Render](https://img.shields.io/badge/Deployment-Render-46E3B7?style=for-the-badge&logo=render)

---

### 🧠 Problem

Flood-prone regions like Assam lack accessible tools to quickly understand district-level flood risk and impact.

---

### 💡 Solution

A web dashboard combining ML-based risk classification with an interactive map to provide real-time flood intelligence and actionable insights.

---

### ✨ Features

- District-level flood risk classification (Low / Moderate / High)  
- Interactive map with real-time visualization  
- 3-day river level forecasts  
- Crop damage estimation  
- Batch prediction across all districts  

---

### 🏗️ Architecture

Client (Leaflet UI) → Flask API → ML Engine → Data Pipeline

---

### ⚙️ Setup

```bash
git clone https://github.com/your-username/flood-dashboard-assam.git
cd flood-dashboard-assam

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python app.py
