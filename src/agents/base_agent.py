class BaseAgent:
    def __init__(self, name):
        self.name = name

    def act(self, observations):
        raise NotImplementedError("Each agent must implement its own 'act' method.")
