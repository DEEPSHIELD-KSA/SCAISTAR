## ⚽ SCAISTAR - AI Football Assistant

SCAISTAR is an AI-powered football assistant built with Streamlit. It delivers real-time match data, player stats, global news, and smart insights using GPT-4o.

### 🔑 Features  
- 💬 **AI Chat** – Ask any football-related question  
- 🔴 **Live Matches** – Real-time scores & match info  
- 🎮 **Player Stats** – Search & compare players  
- 📰 **News** – Latest football headlines  
- 📈 **Analytics** – League tables & predictions  

### 🚀 Quick Start  
1. Clone the repo  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Add your OpenAI API key in `initialize_clients()`  
4. Run the app:  
   ```bash  
   streamlit run app.py  
   ```

### 🔐 API Keys  
Only OpenAI key is required. Add it in `initialize_clients()`:
```python
openai_api_key = "your-openai-api-key"  
os.environ["OPENAI_API_KEY"] = openai_api_key
```

### 📦 Dependencies  
`streamlit`, `requests`, `pandas`, `plotly`, `openai`, `scikit-learn`, `pillow`, `numpy`

