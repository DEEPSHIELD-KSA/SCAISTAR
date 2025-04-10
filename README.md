Overview
SCAISTAR is an AI-powered football assistant built with Streamlit that provides real-time match data, player statistics, news, and intelligent analysis using OpenAI's GPT-4o.
Features

💬 AI Chat Assistant - Ask any football question and get intelligent answers
🔴 Live Matches - Real-time scores and match details
🎮 Player Database - Search and compare player statistics
📰 News - Latest football news from around the world
📈 Analytics - League tables, team analysis, and predictions

Quick Start

Clone the repository
Install dependencies:
pip install -r requirements.txt

Add your OpenAI API key in the initialize_clients() function
Run the application:
streamlit run app.py


API Keys
This app requires an OpenAI API key. Other APIs (RapidAPI, News API) are pre-configured.
To add your OpenAI API key:
pythondef initialize_clients():
    openai_api_key = "your-openai-api-key"  # <-- Replace with your key
    os.environ["OPENAI_API_KEY"] = openai_api_key
    client = OpenAI(api_key=openai_api_key)

    
Dependencies

streamlit
requests
pandas
plotly
openai
scikit-learn
pillow
numpy
