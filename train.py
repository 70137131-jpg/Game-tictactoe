import time
from tictactoe.agent import Qlearner
from tictactoe.teacher import Teacher
from tictactoe.game import Game

def train_agent(episodes=50000, teacher_ability=0.9):
    """Train a Q-learning agent."""
    print(f"Training Q-learning agent for {episodes} episodes...")
    print(f"Teacher ability level: {teacher_ability}")
    
    agent = Qlearner(alpha=0.5, gamma=0.9, eps=0.1)
    teacher = Teacher(level=teacher_ability)
    
    start_time = time.time()
    
    for i in range(episodes):
        game = Game(agent, teacher=teacher)
        game.start()
        
        # Progress updates
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {i + 1}/{episodes} | Time: {elapsed:.1f}s")
    
    # Saves trained agent
    agent.save('q_agent.pkl')
    
    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.1f} seconds!")
    print(f"Agent saved to: q_agent.pkl")
    print(f"Total rewards: {sum(agent.rewards)}")
    print(f"Win rate (approx): {(sum(1 for r in agent.rewards if r > 0) / len(agent.rewards)) * 100:.1f}%")

if __name__ == "__main__":
    train_agent(episodes=50000, teacher_ability=0.9)
