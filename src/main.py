from agents.core.experimental_agent import ExperimentalAgent
from agents.core.theoretical_agent import TheoreticalAgent
# from environments.physics_lab import PhysicsLab
from communication.protocol import CommunicationAgent
# from experiments.pendulum_discovery import run_pendulum_experiment

def main():
    lab = PhysicsLab()

    agent1 = ExperimentalAgent(name="ExpAgent-1")
    agent2 = TheoreticalAgent(name="TheoAgent-1")

    protocol = CommunicationAgent([agent1, agent2])

    # 4. Run experiment
    print("🔬 Starting Pendulum Discovery Experiment...")
    run_pendulum_experiment(lab, [agent1, agent2], protocol)


if __name__ == "__main__":
    main()
