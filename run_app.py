import os
import sys

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Tic-Tac-Toe RL Flask Server")
    print("=" * 50)
    
    # Check if agent exists
    agent_path = 'q_agent.pkl'
    if not os.path.isfile(agent_path):
        print(f"\n⚠️  WARNING: No trained agent found at {agent_path}")
        print("The agent will start with zero training.")
        print("You can train it via the web UI by clicking 'Train 1000 episodes'")
        print("or run: python play.py -a q -t 50000\n")
    else:
        print(f"\n✅ Found trained agent: {agent_path}\n")
    
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    print("=" * 50 + "\n")
    
    # Import and run Flask app
    from app import app
    app.run(host='0.0.0.0', port=5000, debug=True)
