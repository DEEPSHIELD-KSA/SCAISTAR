import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from PIL import Image
import base64

# Page Configuration
st.set_page_config(
    page_title="SCAISTAR",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the UI
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #091747;
            color: white;
            border-radius: 4px 4px 0 0;
            padding-left: 15px;
            padding-right: 15px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3366ff !important;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
        }
        .chat-message.user {
            background-color: #e6f3ff;
        }
        .chat-message.assistant {
            background-color: #f0f0f0;
        }
        .stButton>button {
            background-color: #3366ff;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .sidebar .sidebar-content {
            background-color: #091747;
            color: white;
        }
        h1, h2, h3 {
            color: #091747;
        }
        .stAlert {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: white;
        }
        .news-card {
            border-left: 5px solid #3366ff;
            padding-left: 10px;
            margin-bottom: 15px;
            background-color: white;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .match-card {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .team {
            text-align: center;
            width: 40%;
        }
        .score {
            font-weight: bold;
            font-size: 24px;
            text-align: center;
            width: 20%;
        }
        .team-logo {
            width: 50px;
            height: 50px;
            margin: 0 auto;
            display: block;
        }
        .status-badge {
            background-color: #28a745;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 12px;
            display: inline-block;
        }
        .player-card {
            display: flex;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .player-info {
            margin-left: 15px;
        }
        .player-photo {
            width: 80px;
            height: 80px;
            border-radius: 40px;
        }
        .spinner {
            text-align: center;
            margin: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply CSS
local_css()

# Initialize API credentials and clients
def initialize_clients():
    # Hardcode your OpenAI API key directly in the code
    openai_api_key = ""  # <-- Replace this with your key.
    os.environ["OPENAI_API_KEY"] = openai_api_key
    client = OpenAI(api_key=openai_api_key)

    # API endpoints and headers
    SOFASCORE_URL = "https://www.sofascore.com/api/v1/sport/football/events/live"
    RAPIDAPI_URL = "https://free-api.live-football-data.p.rapidapi.com/football/players-search"
    RAPIDAPI_KEY = "a6f2f2e6b6msh3e7d0724a4c0c1f85b5d1d90007365"
    NEWS_API_URL = "https://newsapi.org/v2/top-headlines"
    NEWS_API_KEY = "eb115385d5744ce297b9d145c61f4ada"
    NEWS_API_PARAMS = {
        "category": "sports",
        "language": "en",
        "apiKey": NEWS_API_KEY
    }

    return {
        "openai_client": client,
        "sofascore_url": SOFASCORE_URL,
        "rapidapi_url": RAPIDAPI_URL,
        "rapidapi_key": RAPIDAPI_KEY,
        "news_api_url": NEWS_API_URL,
        "news_api_params": NEWS_API_PARAMS
    }

# Cache decorator for API calls to avoid repeated calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data_from_sofascore(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Error fetching SofaScore data: {e}")
        return {"events": []}

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_player_data(url, headers, params):
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Error fetching player data: {e}")
        return {}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_sports_news(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f"Error fetching sports news: {e}")
        return {"articles": []}

# Function to process the user's question using OpenAI
def get_ai_response(client, question, context_data):
    try:
        prompt = f"""
        You are an expert sports analyst. Answer the following question
        based on the provided live sports data. Be concise, informative, and engaging.

        Today's date: {datetime.now().strftime('%Y-%m-%d')}

        User question: {question}

        Available context data:
        {json.dumps(context_data, indent=2)}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert sports analyst specializing in football."},
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {e}"

# Function to create football match statistics chart
def create_match_stats_chart(team1="Team A", team2="Team B"):
    # Sample data - would be replaced with real data in production
    categories = ['Possession', 'Shots', 'Shots on Target', 'Corners', 'Fouls']
    team1_values = [65, 12, 5, 8, 10]
    team2_values = [35, 8, 2, 4, 12]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=team1_values,
        theta=categories,
        fill='toself',
        name=team1,
        line_color='#3366ff'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=team2_values,
        theta=categories,
        fill='toself',
        name=team2,
        line_color='#ff3366'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(team1_values), max(team2_values)) + 5]
            )
        ),
        showlegend=True,
        title=f"Match Statistics: {team1} vs {team2}"
    )
    
    return fig

# Function to create player comparison chart
def create_player_comparison(player1="Player 1", player2="Player 2"):
    # Sample data - would be replaced with real data in production
    categories = ['Goals', 'Assists', 'Pass Accuracy', 'Tackles', 'Interceptions']
    player1_values = [12, 8, 85, 45, 32]
    player2_values = [15, 6, 78, 52, 38]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=categories,
        y=player1_values,
        name=player1,
        marker_color='#3366ff'
    ))
    
    fig.add_trace(go.Bar(
        x=categories,
        y=player2_values,
        name=player2,
        marker_color='#ff3366'
    ))
    
    fig.update_layout(
        title=f"Player Comparison: {player1} vs {player2}",
        xaxis_title="Metrics",
        yaxis_title="Values",
        barmode='group'
    )
    
    return fig

# Function to create team form chart
def create_team_form_chart(team="Team A"):
    # Sample data - would be replaced with real data in production
    last_matches = ['W', 'L', 'W', 'W', 'D', 'L', 'W', 'W', 'D', 'W']
    match_points = []
    cumulative_points = []
    total = 0
    
    for result in last_matches:
        if result == 'W':
            points = 3
        elif result == 'D':
            points = 1
        else:
            points = 0
        match_points.append(points)
        total += points
        cumulative_points.append(total)
    
    matches = [f"Match {i+1}" for i in range(len(last_matches))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=matches,
        y=cumulative_points,
        mode='lines+markers',
        name='Cumulative Points',
        line=dict(color='#3366ff', width=3),
        marker=dict(size=10)
    ))
    
    # Add color-coded points for each match result
    for i, result in enumerate(last_matches):
        color = '#28a745' if result == 'W' else '#ffc107' if result == 'D' else '#dc3545'
        fig.add_trace(go.Scatter(
            x=[matches[i]],
            y=[cumulative_points[i]],
            mode='markers',
            marker=dict(
                color=color,
                size=12,
                line=dict(
                    color='white',
                    width=2
                )
            ),
            showlegend=False,
            hovertemplate=
            f'Match {i+1}<br>' +
            f'Result: {result}<br>' +
            f'Points: {match_points[i]}<br>' +
            f'Total: {cumulative_points[i]}'
        ))
    
    fig.update_layout(
        title=f"{team} - Form in Last 10 Matches",
        xaxis_title="Matches",
        yaxis_title="Points",
        hovermode="closest"
    )
    
    return fig

# Function to create league standings table
def create_league_standings():
    # Sample data - would be replaced with real data in production
    data = {
        'Position': list(range(1, 11)),
        'Team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 
                 'Team F', 'Team G', 'Team H', 'Team I', 'Team J'],
        'Played': [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        'Won': [8, 7, 6, 6, 5, 5, 4, 3, 2, 1],
        'Drawn': [1, 2, 2, 1, 3, 2, 2, 3, 3, 2],
        'Lost': [1, 1, 2, 3, 2, 3, 4, 4, 5, 7],
        'GF': [25, 23, 18, 17, 15, 14, 12, 10, 8, 6],
        'GA': [8, 9, 10, 12, 11, 13, 15, 16, 18, 22],
        'GD': [17, 14, 8, 5, 4, 1, -3, -6, -10, -16],
        'Points': [25, 23, 20, 19, 18, 17, 14, 12, 9, 5]
    }
    
    df = pd.DataFrame(data)
    
    # Format the table with styling
    styled_df = df.style.apply(lambda x: ['background-color: #e6f3ff' if i % 2 == 0 else '' 
                                         for i in range(len(x))], axis=0)
    
    styled_df = styled_df.set_properties(**{
        'text-align': 'center',
        'font-size': '14px',
        'border': '1px solid #ddd',
        'padding': '8px'
    })
    
    return styled_df

# Create header with logo
def create_header():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #091747; font-size: 40px; margin-bottom: 0;">SCAISTAR</h1>
            <p style="color: #666; font-size: 18px;">Your professional sports assistant powered by AI</p>
        </div>
        """, unsafe_allow_html=True)

# Create sidebar for settings and info (Removed API key input)
def create_sidebar():
    with st.sidebar:
        st.title("SCAISTAR")
        
        # Removed the API key settings block so the key is now directly in the code.
        st.subheader("📊 Features")
        st.markdown("""
        - 💬 **AI Chat Assistant** - Ask any football question
        - 🔴 **Live Matches** - Real-time scores and stats
        - 🏆 **League Tables** - Standings and team performance
        - 🎮 **Player Database** - Search and compare players
        - 📰 **Sports News** - Latest updates and articles
        - 📈 **Match Analytics** - Advanced statistics and predictions
        """)
        
        st.markdown("---")
        
        st.subheader("ℹ️ About")
        st.info("""
       SCAISTAR is your personal football assistant.
        
        Get real-time match data, player information, stats, and intelligent answers to all your football questions!
        """)

# Home tab content
def display_home():
    st.markdown("""
    <div class="card">
        <h2>👋 Welcome to SCAISTAR!</h2>
        <p>Your all-in-one platform for football insights, stats, and analysis powered by artificial intelligence.</p>
        <p>Navigate through the tabs above to explore different features:</p>
        <ul>
            <li><strong>Chat Assistant</strong>: Ask any football-related question</li>
            <li><strong>Live Matches</strong>: View real-time scores and match details</li>
            <li><strong>Players</strong>: Search and analyze player statistics</li>
            <li><strong>News</strong>: Latest football news from around the world</li>
            <li><strong>Analytics</strong>: Advanced statistics and visualizations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔥 Featured Match")
        match_data = {
            "home_team": "Arsenal FC",
            "away_team": "Manchester City",
            "home_score": 2,
            "away_score": 2,
            "time": "Live - 78'",
            "league": "Premier League"
        }
        
        st.markdown(f"""
        <div class='match-card'>
            <div class='team'>
                <img src='https://img.icons8.com/color/48/000000/arsenal-fc.png' class='team-logo'>
                <p>{match_data['home_team']}</p>
            </div>
            <div class='score'>
                <span class='status-badge'>{match_data['time']}</span>
                <p>{match_data['home_score']} - {match_data['away_score']}</p>
            </div>
            <div class='team'>
                <img src='https://img.icons8.com/color/48/000000/manchester-city.png' class='team-logo'>
                <p>{match_data['away_team']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📰 Latest News")
        st.markdown("""
        <div class="news-card">
            <h4>Premier League announces new broadcast deal</h4>
            <p>The Premier League has announced a record-breaking new broadcast deal worth £5 billion...</p>
            <small>2 hours ago</small>
        </div>
        <div class="news-card">
            <h4>Champions League quarter-finals set to begin</h4>
            <p>Eight teams remain in the hunt for European glory as the Champions League...</p>
            <small>5 hours ago</small>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Top Performers This Week")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <img src="https://img.icons8.com/color/96/000000/football-player.png" width="50">
                <h4>Top Scorer</h4>
                <p>Erling Haaland</p>
                <p style="font-size: 24px; font-weight: bold;">3 goals</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown("""
            <div style="text-align: center; padding: 10px;">
                <img src="https://img.icons8.com/color/96/000000/football-team.png" width="50">
                <h4>Top Assists</h4>
                <p>Kevin De Bruyne</p>
                <p style="font-size: 24px; font-weight: bold;">4 assists</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💡 Ask SCAISTAR")
        quick_questions = [
            "Who's likely to win the Premier League this season?",
            "Which player has the most goals in Champions League?",
            "Show me the best young talents to watch",
            "Analyze Arsenal's defensive performance"
        ]
        
        for q in quick_questions:
            if st.button(q):
                st.session_state.question = q
                st.session_state.tab = "Chat"
        
        st.markdown("</div>", unsafe_allow_html=True)

# Chat tab content
def display_chat(clients):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 AI Chat Assistant")
    st.markdown("Ask me anything about football - matches, players, statistics, predictions, and more!")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Handle pre-filled question from quick questions
    if "question" in st.session_state:
        prompt = st.session_state.question
        st.session_state.pop("question")
    else:
        # Input for new question
        prompt = st.chat_input("Ask me about football/soccer...")
    
    if prompt:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show assistant response with loading indicator
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Analyzing football data..."):
                # Get context data for AI response
                context = {
                    "live_matches": fetch_data_from_sofascore(clients["sofascore_url"]),
                    "sports_news": fetch_sports_news(clients["news_api_url"], clients["news_api_params"]).get("articles", [])[:5]
                }
                
                # Generate AI response
                response = get_ai_response(clients["openai_client"], prompt, context)
                
                # Display response
                message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Live matches tab content
def display_live_matches(clients):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔴 Live Matches")
    st.markdown("Real-time scores and match information from around the world")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Fetch live match data
    with st.spinner("Fetching live match data..."):
        live_data = fetch_data_from_sofascore(clients["sofascore_url"])
    
    # Sample data for demonstration (would use real API data in production)
    sample_matches = [
        {
            "home_team": "Arsenal FC",
            "away_team": "Manchester City",
            "home_score": 2,
            "away_score": 2,
            "time": "Live - 78'",
            "league": "Premier League"
        },
        {
            "home_team": "Barcelona",
            "away_team": "Real Madrid",
            "home_score": 1,
            "away_score": 1,
            "time": "Live - 62'",
            "league": "La Liga"
        },
        {
            "home_team": "Bayern Munich",
            "away_team": "Borussia Dortmund",
            "home_score": 3,
            "away_score": 1,
            "time": "Live - 81'",
            "league": "Bundesliga"
        },
        {
            "home_team": "Liverpool",
            "away_team": "Manchester United",
            "home_score": 2,
            "away_score": 0,
            "time": "HT",
            "league": "Premier League"
        }
    ]
    
    # Display filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        league_filter = st.selectbox("Filter by League", 
                                     ["All Leagues", "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
    with col2:
        status_filter = st.selectbox("Filter by Status", 
                                    ["All Statuses", "Live", "Not Started", "Finished"])
    with col3:
        team_filter = st.text_input("Search by Team")
    
    # Create tabs for different match categories
    match_tabs = st.tabs(["Live Now", "Today", "Upcoming", "Finished"])
    
    with match_tabs[0]:
        for match in sample_matches:
            st.markdown(f"""
            <div class='match-card'>
                <div class='team'>
                    <img src='https://img.icons8.com/color/48/000000/football-team.png' class='team-logo'>
                    <p>{match['home_team']}</p>
                </div>
                <div class='score'>
                    <span class='status-badge'>{match['time']}</span>
                    <p>{match['home_score']} - {match['away_score']}</p>
                    <small>{match['league']}</small>
                </div>
                <div class='team'>
                    <img src='https://img.icons8.com/color/48/000000/football-team.png' class='team-logo'>
                    <p>{match['away_team']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a button to view detailed match statistics
            if st.button(f"View Stats: {match['home_team']} vs {match['away_team']}"):
                st.session_state['selected_match'] = match
                st.session_state['show_match_details'] = True
    
    # Display match details if a match is selected
    if st.session_state.get('show_match_details', False):
        match = st.session_state['selected_match']
        st.markdown(f"<h3>{match['home_team']} vs {match['away_team']}</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Match Information")
            st.markdown(f"""
            <p><strong>Competition:</strong> {match['league']}</p>
            <p><strong>Stadium:</strong> Emirates Stadium, London</p>
            <p><strong>Referee:</strong> Michael Oliver</p>
            <p><strong>Attendance:</strong> 60,352</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Match Events")
            st.markdown("""
            <p>⚽ <strong>12'</strong> Goal - B. Saka (Arsenal)</p>
            <p>🟨 <strong>23'</strong> Yellow Card - R. Dias (Man City)</p>
            <p>⚽ <strong>34'</strong> Goal - K. De Bruyne (Man City)</p>
            <p>⚽ <strong>52'</strong> Goal - G. Martinelli (Arsenal)</p>
            <p>🔄 <strong>65'</strong> Substitution - J. Grealish IN, P. Foden OUT (Man City)</p>
            <p>⚽ <strong>71'</strong> Goal - E. Haaland (Man City)</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Display match statistics chart
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Match Statistics")
            fig = create_match_stats_chart(match['home_team'], match['away_team'])
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add lineup visualization
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Team Lineups")
            lineup_tab1, lineup_tab2 = st.tabs([match['home_team'], match['away_team']])
            
            with lineup_tab1:
                st.markdown("""
                <p style="font-weight: bold;">Formation: 4-3-3</p>
                <p>1. Ramsdale (GK)</p>
                <p>4. White - 6. Gabriel - 5. Holding - 3. Tierney</p>
                <p>8. Ødegaard - 5. Partey - 34. Xhaka</p>
                <p>7. Saka - 9. Jesus - 11. Martinelli</p>
                """, unsafe_allow_html=True)
            
            with lineup_tab2:
                st.markdown("""
                <p style="font-weight: bold;">Formation: 4-2-3-1</p>
                <p>31. Ederson (GK)</p>
                <p>2. Walker - 5. Stones - 14. Laporte - 7. Cancelo</p>
                <p>16. Rodri - 17. De Bruyne</p>
                <p>26. Mahrez - 20. Silva - 47. Foden</p>
                <p>9. Haaland</p>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# Players tab content
def display_players(clients):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎮 Player Database")
    st.markdown("Search for players, view stats, and compare performances")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Search functionality
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        player_search = st.text_input("Search for a player by name", "")
    with search_col2:
        search_button = st.button("Search")
    
    # If search button is clicked or player name is entered
    if search_button or player_search:
        with st.spinner("Searching for player..."):
            # Prepare for API call (in real implementation)
            headers = {"X-RapidAPI-Key": clients["rapidapi_key"]}
            params = {"search": player_search}
            
            # For demonstration, use sample data
            sample_players = [
                {
                    "name": "Lionel Messi",
                    "age": 35,
                    "team": "Inter Miami CF",
                    "position": "Forward",
                    "nationality": "Argentina",
                    "image": "https://img.icons8.com/color/96/000000/messi.png"
                },
                {
                    "name": "Cristiano Ronaldo",
                    "age": 38,
                    "team": "Al-Nassr FC",
                    "position": "Forward",
                    "nationality": "Portugal",
                    "image": "https://img.icons8.com/color/96/000000/cristiano-ronaldo.png"
                }
            ]
            
            # Filter sample data based on search
            filtered_players = [p for p in sample_players if player_search.lower() in p["name"].lower()]
            
            if filtered_players:
                for player in filtered_players:
                    st.markdown(f"""
                    <div class='player-card'>
                        <img src='{player['image']}' class='player-photo' alt='{player['name']}'>
                        <div class='player-info'>
                            <h3>{player['name']}</h3>
                            <p><strong>Team:</strong> {player['team']}</p>
                            <p><strong>Position:</strong> {player['position']}</p>
                            <p><strong>Age:</strong> {player['age']}</p>
                            <p><strong>Nationality:</strong> {player['nationality']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"View detailed stats for {player['name']}"):
                        st.session_state['selected_player'] = player
                        st.session_state['show_player_details'] = True
            else:
                st.warning(f"No players found matching '{player_search}'")
    
    # Player comparison section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Player Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Select first player", ["Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappé", "Erling Haaland"])
    with col2:
        player2 = st.selectbox("Select second player", ["Cristiano Ronaldo", "Lionel Messi", "Kylian Mbappé", "Erling Haaland"])
    
    if st.button("Compare Players"):
        fig = create_player_comparison(player1, player2)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display player details if a player is selected
    if st.session_state.get('show_player_details', False):
        player = st.session_state['selected_player']
        
        st.markdown(f"<h3>{player['name']} - Detailed Statistics</h3>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Match History", "Career Stats"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(player['image'], width=150)
                st.markdown(f"""
                <div style="background-color: white; padding: 15px; border-radius: 5px;">
                    <p><strong>Name:</strong> {player['name']}</p>
                    <p><strong>Age:</strong> {player['age']}</p>
                    <p><strong>Team:</strong> {player['team']}</p>
                    <p><strong>Position:</strong> {player['position']}</p>
                    <p><strong>Nationality:</strong> {player['nationality']}</p>
                    <p><strong>Height:</strong> 1.70m</p>
                    <p><strong>Weight:</strong> 72kg</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("2024/25 Season Statistics")
                # Create sample stats
                stats = {
                    "Games": 28,
                    "Goals": 22,
                    "Assists": 14,
                    "Minutes Played": 2340,
                    "Yellow Cards": 2,
                    "Red Cards": 0
                }
                
                # Create a radar chart for player attributes
                attributes = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]
                values = [95, 97, 98, 99, 40, 68]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=attributes,
                    fill='toself',
                    name=player['name'],
                    line_color='#3366ff'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=False,
                    title="Player Attributes"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create stats table
                stats_df = pd.DataFrame(list(stats.items()), columns=["Stat", "Value"])
                st.table(stats_df)
        
        with tab2:
            st.subheader("Recent Matches")
            
            matches = [
                {"date": "2024-04-03", "competition": "Champions League", "opponent": "Bayern Munich", "result": "W 3-1", "goals": 2, "assists": 1},
                {"date": "2024-03-28", "competition": "La Liga", "opponent": "Real Madrid", "result": "D 2-2", "goals": 1, "assists": 0},
                {"date": "2024-03-21", "competition": "La Liga", "opponent": "Valencia", "result": "W 4-0", "goals": 2, "assists": 2},
                {"date": "2024-03-17", "competition": "Champions League", "opponent": "Manchester City", "result": "W 2-1", "goals": 1, "assists": 1},
                {"date": "2024-03-10", "competition": "La Liga", "opponent": "Sevilla", "result": "W 3-0", "goals": 1, "assists": 0}
            ]
            
            match_df = pd.DataFrame(matches)
            st.dataframe(match_df, use_container_width=True)
            
            match_nums = list(range(1, len(matches) + 1))
            goals = [match["goals"] for match in matches]
            assists = [match["assists"] for match in matches]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=match_nums,
                y=goals,
                mode='lines+markers',
                name='Goals',
                line=dict(color='#3366ff', width=3),
                marker=dict(size=10)
            ))
            fig.add_trace(go.Scatter(
                x=match_nums,
                y=assists,
                mode='lines+markers',
                name='Assists',
                line=dict(color='#ff3366', width=3),
                marker=dict(size=10)
            ))
            fig.update_layout(
                title="Performance in Last 5 Matches",
                xaxis_title="Match Number",
                yaxis_title="Count",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Career Statistics")
            
            career = [
                {"season": "2023/24", "team": "Inter Miami CF", "games": 32, "goals": 28, "assists": 18},
                {"season": "2022/23", "team": "PSG", "games": 41, "goals": 21, "assists": 20},
                {"season": "2021/22", "team": "PSG", "games": 34, "goals": 11, "assists": 15},
                {"season": "2020/21", "team": "Barcelona", "games": 47, "goals": 38, "assists": 14},
                {"season": "2019/20", "team": "Barcelona", "games": 44, "goals": 31, "assists": 27},
                {"season": "2018/19", "team": "Barcelona", "games": 50, "goals": 51, "assists": 22}
            ]
            
            career_df = pd.DataFrame(career)
            st.dataframe(career_df, use_container_width=True)
            
            seasons = [c["season"] for c in career]
            goals = [c["goals"] for c in career]
            assists = [c["assists"] for c in career]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=seasons,
                y=goals,
                name='Goals',
                marker_color='#3366ff'
            ))
            fig.add_trace(go.Bar(
                x=seasons,
                y=assists,
                name='Assists',
                marker_color='#ff3366'
            ))
            fig.update_layout(
                title="Goals and Assists by Season",
                xaxis_title="Season",
                yaxis_title="Count",
                barmode='group',
                xaxis={'categoryorder':'array', 'categoryarray':seasons[::-1]}
            )
            
            st.plotly_chart(fig, use_container_width=True)

# News tab content
def display_news(clients):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📰 Football News")
    st.markdown("Latest news, updates, and articles from the world of football")
    st.markdown("</div>", unsafe_allow_html=True)
    
    with st.spinner("Fetching latest news..."):
        news_data = fetch_sports_news(clients["news_api_url"], clients["news_api_params"])
    
    sample_news = [
        {
            "title": "Premier League announces new broadcast deal",
            "description": "The Premier League has announced a record-breaking new broadcast deal worth £5 billion for the next three seasons.",
            "source": {"name": "BBC Sport"},
            "publishedAt": "2024-04-10T10:30:00Z",
            "urlToImage": "https://via.placeholder.com/300x200?text=Premier+League"
        },
        {
            "title": "Champions League quarter-finals set to begin",
            "description": "Eight teams remain in the hunt for European glory as the Champions League quarter-finals kick off this week.",
            "source": {"name": "Sky Sports"},
            "publishedAt": "2024-04-09T15:45:00Z",
            "urlToImage": "https://via.placeholder.com/300x200?text=Champions+League"
        },
        {
            "title": "Messi scores hat-trick in Inter Miami's victory",
            "description": "Lionel Messi continued his exceptional form with a hat-trick in Inter Miami's 5-0 win in the MLS.",
            "source": {"name": "ESPN"},
            "publishedAt": "2024-04-08T08:20:00Z",
            "urlToImage": "https://via.placeholder.com/300x200?text=Messi"
        },
        {
            "title": "England announce squad for upcoming internationals",
            "description": "Gareth Southgate has named his England squad for the upcoming international fixtures, with several surprise inclusions.",
            "source": {"name": "The Guardian"},
            "publishedAt": "2024-04-07T13:10:00Z",
            "urlToImage": "https://via.placeholder.com/300x200?text=England"
        },
        {
            "title": "Juventus sack manager after poor run of results",
            "description": "Juventus have parted ways with their manager following a disappointing run of results in Serie A.",
            "source": {"name": "Goal.com"},
            "publishedAt": "2024-04-06T09:40:00Z",
            "urlToImage": "https://via.placeholder.com/300x200?text=Juventus"
        }
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        source_filter = st.selectbox("Filter by Source", 
                                    ["All Sources", "BBC Sport", "Sky Sports", "ESPN", "The Guardian", "Goal.com"])
    with col2:
        topic_filter = st.selectbox("Filter by Topic", 
                                   ["All Topics", "Premier League", "Champions League", "International", "Transfers"])
    
    for i, news in enumerate(sample_news):
        st.markdown(f"""
        <div class="news-card">
            <div style="display: flex; gap: 20px;">
                <div style="flex: 1;">
                    <img src="{news['urlToImage']}" style="width: 100%; border-radius: 5px;">
                </div>
                <div style="flex: 3;">
                    <h3>{news['title']}</h3>
                    <p>{news['description']}</p>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <span style="color: #666;">{news['source']['name']}</span>
                        <span style="color: #666;">{news['publishedAt'].split('T')[0]}</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Trending Topics")
    topics = ["#ChampionsLeague", "#PremierLeague", "#Messi", "#Ronaldo", "#TransferNews"]
    st.markdown(
        '<div style="display: flex; gap: 10px; flex-wrap: wrap;">'+
        ''.join([f'<div style="background-color: #e6f3ff; padding: 8px 15px; border-radius: 20px;">{topic}</div>' for topic in topics])+
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Analytics tab content
def display_analytics():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Match Analytics")
    st.markdown("Advanced statistics, visualizations, and predictions")
    st.markdown("</div>", unsafe_allow_html=True)
    
    analytics_tabs = st.tabs(["League Tables", "Team Analysis", "Predictions", "Advanced Stats"])
    
    with analytics_tabs[0]:
        league_selector = st.selectbox("Select League", 
                                      ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
        
        st.subheader(f"{league_selector} Standings")
        standings_df = create_league_standings()
        st.dataframe(standings_df, hide_index=True)
        
        st.subheader("League Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Top Scorers")
            top_scorers = {
                "Player": ["E. Haaland", "H. Kane", "M. Salah", "S. Son", "K. Mbappé"],
                "Team": ["Man City", "Bayern", "Liverpool", "Tottenham", "Real Madrid"],
                "Goals": [27, 24, 21, 19, 18]
            }
            scorers_df = pd.DataFrame(top_scorers)
            fig = px.bar(scorers_df, x="Player", y="Goals", 
                        text="Goals", color="Team", 
                        title="Top Goal Scorers")
            fig.update_layout(xaxis_title="", yaxis_title="Goals Scored")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Top Assists")
            top_assists = {
                "Player": ["K. De Bruyne", "B. Fernandes", "T. Kroos", "L. Messi", "R. Kvaratskhelia"],
                "Team": ["Man City", "Man United", "Real Madrid", "Inter Miami", "Napoli"],
                "Assists": [18, 15, 14, 14, 12]
            }
            assists_df = pd.DataFrame(top_assists)
            fig = px.bar(assists_df, x="Player", y="Assists", 
                        text="Assists", color="Team", 
                        title="Top Assists Providers")
            fig.update_layout(xaxis_title="", yaxis_title="Assists Made")
            st.plotly_chart(fig, use_container_width=True)
    
    with analytics_tabs[1]:
        team_selector = st.selectbox("Select Team", 
                                    ["Arsenal", "Manchester City", "Liverpool", "Barcelona", "Real Madrid"])
        st.subheader(f"{team_selector} - Team Analysis")
        st.markdown("#### Form in Last 10 Matches")
        form_chart = create_team_form_chart(team_selector)
        st.plotly_chart(form_chart, use_container_width=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Attacking Performance")
            attacking_stats = {
                "Metric": ["Goals Scored", "Shots per Game", "Possession %", "Pass Accuracy", "Crosses per Game"],
                "Value": [52, 16.8, 65.3, 88.5, 22.3],
                "League Rank": ["1st", "2nd", "1st", "1st", "3rd"]
            }
            attack_df = pd.DataFrame(attacking_stats)
            st.dataframe(attack_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Defensive Performance")
            defensive_stats = {
                "Metric": ["Goals Conceded", "Clean Sheets", "Tackles per Game", "Interceptions", "Aerial Duels Won %"],
                "Value": [24, 12, 18.5, 9.2, 62.7],
                "League Rank": ["2nd", "3rd", "5th", "4th", "6th"]
            }
            defense_df = pd.DataFrame(defensive_stats)
            st.dataframe(defense_df, hide_index=True, use_container_width=True)
        
        st.markdown("#### Team Strengths & Weaknesses")
        strength_weak_cols = st.columns(2)
        with strength_weak_cols[0]:
            st.markdown("""
            <div style="background-color: #d4edda; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <h5 style="color: #155724;">Strengths</h5>
                <ul>
                    <li>Excellent ball possession (65.3% - 1st in league)</li>
                    <li>Strong attacking output (52 goals - 1st in league)</li>
                    <li>High defensive line enabling territorial dominance</li>
                    <li>Exceptional passing accuracy (88.5% - 1st in league)</li>
                    <li>Creative midfield generating high-quality chances</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with strength_weak_cols[1]:
            st.markdown("""
            <div style="background-color: #f8d7da; padding: 15px; border-radius: 5px; margin-top: 10px;">
                <h5 style="color: #721c24;">Weaknesses</h5>
                <ul>
                    <li>Vulnerable to counter-attacks</li>
                    <li>Inconsistent aerial duel success (62.7% - 6th in league)</li>
                    <li>Relies heavily on key players</li>
                    <li>Struggles against low defensive blocks</li>
                    <li>Defensive concentration in final 15 minutes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with analytics_tabs[2]:
        st.markdown("#### Match Outcome Predictions")
        upcoming_matches = [
            {"home": "Arsenal", "away": "Liverpool", "competition": "Premier League", "date": "2024-04-15"},
            {"home": "Barcelona", "away": "Real Madrid", "competition": "La Liga", "date": "2024-04-21"},
            {"home": "Bayern Munich", "away": "Borussia Dortmund", "competition": "Bundesliga", "date": "2024-04-18"}
        ]
        for match in upcoming_matches:
            st.markdown(f"""
            <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                <h5>{match['home']} vs {match['away']} - {match['competition']}</h5>
                <p>Date: {match['date']}</p>
                <div style="display: flex; margin-top: 10px; gap: 10px;">
                    <div style="flex: 1; text-align: center; background-color: #e6f3ff; padding: 10px; border-radius: 5px;">
                        <h6>{match['home']} Win</h6>
                        <p style="font-size: 24px; font-weight: bold;">45%</p>
                    </div>
                    <div style="flex: 1; text-align: center; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                        <h6>Draw</h6>
                        <p style="font-size: 24px; font-weight: bold;">30%</p>
                    </div>
                    <div style="flex: 1; text-align: center; background-color: #ffe6e6; padding: 10px; border-radius: 5px;">
                        <h6>{match['away']} Win</h6>
                        <p style="font-size: 24px; font-weight: bold;">25%</p>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <p><strong>Score Prediction:</strong> {match['home']} 2-1 {match['away']}</p>
                    <p><strong>Key Factors:</strong> Home advantage, Recent form, Head-to-head record</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Season Outcome Predictions")
        title_race = {
            "Team": ["Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham"],
            "Probability": [45, 35, 15, 3, 2]
        }
        title_df = pd.DataFrame(title_race)
        fig = px.pie(title_df, values="Probability", names="Team", 
                    title="Premier League Title Race Probabilities",
                    color_discrete_sequence=px.colors.sequential.Blues)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with analytics_tabs[3]:
        st.markdown("#### Expected Goals (xG) Analysis")
        xg_data = {
            "Team": ["Man City", "Arsenal", "Liverpool", "Man United", "Chelsea", 
                    "Newcastle", "Tottenham", "Aston Villa", "Brighton", "West Ham"],
            "Goals": [78, 75, 70, 60, 58, 55, 53, 48, 45, 42],
            "xG": [82.3, 68.5, 72.1, 55.8, 61.2, 50.3, 57.5, 45.2, 42.8, 38.5]
        }
        xg_df = pd.DataFrame(xg_data)
        xg_df["Difference"] = xg_df["Goals"] - xg_df["xG"]
        xg_df = xg_df.sort_values("Difference", ascending=False)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=xg_df["Team"],
            y=xg_df["Goals"],
            name="Actual Goals",
            marker_color='#3366ff'
        ))
        fig.add_trace(go.Bar(
            x=xg_df["Team"],
            y=xg_df["xG"],
            name="Expected Goals (xG)",
            marker_color='#ff3366'
        ))
        fig.update_layout(
            title="Goals vs Expected Goals (xG) by Team",
            xaxis_title="Team",
            yaxis_title="Goals",
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Over/Under Performance vs. Expected Goals")
        xg_df["Difference"] = xg_df["Difference"].apply(lambda x: f"+{x:.1f}" if x > 0 else f"{x:.1f}")
        
        def color_difference(val):
            try:
                num = float(val.replace('+', '')) 
                if num > 0:
                    return 'background-color: #d4edda; color: #155724'
                elif num < 0:
                    return 'background-color: #f8d7da; color: #721c24'
                else:
                    return ''
            except:
                return ''
        
        styled_df = xg_df.style.map(color_difference, subset=['Difference'])
        st.dataframe(styled_df, use_container_width=True)
        
        st.markdown("#### Progressive Passing & Carrying Analysis")
        progressive_data = {
            "Player": ["Rodri", "De Bruyne", "Ødegaard", "Fernandes", "Rice", 
                      "Silva", "Eriksen", "Kovacic", "Foden", "Mount"],
            "Progressive Passes": [287, 265, 254, 246, 231, 225, 217, 210, 198, 187],
            "Progressive Carries": [145, 198, 165, 120, 95, 210, 75, 108, 187, 134],
            "Team": ["Man City", "Man City", "Arsenal", "Man Utd", "Arsenal", 
                    "Man City", "Man Utd", "Man City", "Man City", "Man Utd"]
        }
        prog_df = pd.DataFrame(progressive_data)
        fig = px.scatter(prog_df, x="Progressive Passes", y="Progressive Carries", 
                         color="Team", size="Progressive Passes", 
                         hover_name="Player", 
                         title="Progressive Passing vs. Carrying by Player",
                         color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(
            xaxis_title="Progressive Passes",
            yaxis_title="Progressive Carries",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# Settings tab content (Removed API key settings)
def display_settings():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Settings")
    st.markdown("Configure your SCAISTAR experience")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display Settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Display Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        theme = st.selectbox("Theme", ["Light", "Dark", "System Default"])
        date_format = st.selectbox("Date Format", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"])
    with col2:
        time_format = st.selectbox("Time Format", ["24-hour", "12-hour (AM/PM)"])
        temperature_unit = st.selectbox("Temperature Unit", ["Celsius (°C)", "Fahrenheit (°F)"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Notifications")
    st.checkbox("Live Match Alerts", True)
    st.checkbox("Breaking News Notifications", True)
    st.checkbox("Favorite Team Updates", True)
    st.checkbox("Transfer News Alerts", False)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Favorites")
    favorite_tabs = st.tabs(["Teams", "Players", "Competitions"])
    with favorite_tabs[0]:
        st.multiselect("Select Favorite Teams", 
                      ["Arsenal", "Manchester United", "Liverpool", "Barcelona", "Real Madrid", 
                       "Bayern Munich", "Juventus", "PSG", "AC Milan", "Ajax"])
    with favorite_tabs[1]:
        st.multiselect("Select Favorite Players", 
                      ["Lionel Messi", "Cristiano Ronaldo", "Kylian Mbappé", "Erling Haaland", 
                       "Kevin De Bruyne", "Vinícius Júnior", "Jude Bellingham", "Mohamed Salah"])
    with favorite_tabs[2]:
        st.multiselect("Select Favorite Competitions", 
                      ["Premier League", "Champions League", "La Liga", "Bundesliga", "Serie A", 
                       "World Cup", "European Championship", "Copa America"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data Preferences")
    st.slider("Chat History Retention (days)", 1, 30, 7)
    st.checkbox("Allow Analytics Collection", True, help="Helps us improve the application by collecting anonymous usage data")
    st.checkbox("Enable Offline Mode (Save data for offline access)", False)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.button("Save All Settings", type="primary")

# About tab content
def display_about():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About SCAISTAR")
    st.markdown("""
    **SCAISTAR** is your personal football assistant powered by artificial intelligence. 
    Get real-time data, insightful analysis, and engage in natural conversations about your favorite sport.
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Features")
        st.markdown("""
        - **AI Assistant**: Chat with an AI trained on football knowledge
        - **Live Scores**: Real-time match updates from around the world
        - **Player Database**: Comprehensive player statistics and profiles
        - **News Aggregator**: Latest football news from trusted sources
        - **Match Analytics**: Advanced statistics and visual representations
        - **Predictions**: AI-powered match outcome predictions
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Data Sources")
        st.markdown("""
        SCAISTAR aggregates data from various reliable sources including:
        - SofaScore API
        - News API
        - RapidAPI Football Database
        - OpenAI's GPT-4o
        All data is updated in real-time to provide the most accurate information.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("How It Works")
        st.markdown("""
        SCAISTAR combines data from multiple sources and processes it through advanced AI models to provide insightful analysis and answers to your questions.
        1. **Data Collection**: Real-time data from multiple APIs
        2. **AI Processing**: Analysis using OpenAI's GPT-4o
        3. **Visualization**: Interactive charts and tables
        4. **User Interface**: Streamlit-powered responsive design
        The application is constantly learning and improving based on new data and user interactions.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Version & Updates")
        st.markdown("""
        **Current Version**: 2.0.0
        **Latest Updates**:
        - Added advanced match analytics
        - Improved player comparison feature
        - Enhanced UI with tabbed navigation
        - Optimized data loading for faster performance
        - Added dark mode support
        **Coming Soon**:
        - Fantasy football integration
        - Historical match archive
        - Personalized recommendations
        - Mobile app version
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Contact & Support")
    st.markdown("""
    For questions, feedback, or support requests, please contact us:
    **Email**: support@aifancoach.com  
    **Twitter**: @AIFanCoach  
    **Discord**: [Join our community](https://discord.gg/aifancoach)
    We value your feedback and are constantly working to improve the application!
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Main application
def main():
    if "tab" not in st.session_state:
        st.session_state.tab = "Home"
    
    create_header()
    create_sidebar()
    clients = initialize_clients()
    tabs = st.tabs(["🏠 Home", "💬 Chat Assistant", "🔴 Live Matches", "🎮 Players", "📰 News", "📈 Analytics", "⚙️ Settings", "ℹ️ About"])
    
    with tabs[0]:
        display_home()
    with tabs[1]:
        display_chat(clients)
    with tabs[2]:
        display_live_matches(clients)
    with tabs[3]:
        display_players(clients)
    with tabs[4]:
        display_news(clients)
    with tabs[5]:
        display_analytics()
    with tabs[6]:
        display_settings()
    with tabs[7]:
        display_about()

if __name__ == "__main__":
    main()
