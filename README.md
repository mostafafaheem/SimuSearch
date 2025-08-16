# SimuSearch
Multi-Agent Scientific Research Simulator

A sophisticated multi-agent system for conducting scientific research simulations using LangChain and AI-powered agents.

## 🚀 Features

### **Core Agents**
- **Experimental Agent** - Design experiments, collect data, manage equipment
- **Theoretical Agent** - Generate hypotheses, create mathematical models, literature review
- **Communication Agent** - Inter-agent coordination and message routing

### **Key Capabilities**
- **AI-Powered Research** - LangChain integration for intelligent decision making
- **Scientific Workflow** - Complete experiment lifecycle management
- **Multi-Agent Collaboration** - Coordinated research between specialized agents
- **Data Management** - Structured data collection and analysis
- **Reproducibility** - Comprehensive experiment tracking and validation

## 🏗️ Architecture

```
src/
├── agents/
│   ├── core/                    # Core research agents
│   │   ├── experimental_agent.py    # Lab work & data collection
│   │   ├── theoretical_agent.py     # Theory & modeling
│   │   └── base_agent.py           # Base agent class
│   ├── coordination/            # System coordination agents
│   │   ├── communication_agent.py   # Inter-agent messaging
│   │   └── project_manager_agent.py # Task coordination
│   ├── research/               # Specialized research agents
│   │   ├── analysis_agent.py       # Data analysis
│   │   ├── literature_agent.py     # Literature review
│   │   └── validation_agent.py     # Quality assurance
│   └── __init__.py
├── environment/                 # Simulation environment
├── communication/              # Communication protocols
└── experiments/                # Experiment frameworks
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SimuSearch
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🧪 Usage

### **Quick Demo**
```bash
python demo_agents.py
```

### **Full Test Suite**
```bash
python test_agents.py
```

### **Custom Experiment**
```python
from src.agents.core.experimental_agent import ExperimentalAgent
from src.agents.core.theoretical_agent import TheoreticalAgent

# Initialize agents
exp_agent = ExperimentalAgent("ExpAgent-1")
theo_agent = TheoreticalAgent("TheoAgent-1")

# Generate hypothesis
hypothesis = theo_agent.generate_pendulum_hypothesis()

# Design experiment
experiment = exp_agent.design_pendulum_experiment()

# Run experiment workflow
exp_agent.start_experiment("exp_001")
# ... collect data ...
exp_agent.complete_experiment("exp_001")
```

## 🔬 Example: Pendulum Experiment

The system includes a complete pendulum experiment workflow:

1. **Theory Development** - Generate hypothesis about pendulum behavior
2. **Mathematical Modeling** - Create equations for period vs. length
3. **Experiment Design** - Plan data collection methodology
4. **Data Collection** - Measure periods for different lengths
5. **Analysis** - Compare experimental vs. theoretical results
6. **Validation** - Assess hypothesis support

## 🎯 Next Steps

- [ ] Implement Analysis Agent for statistical analysis
- [ ] Build Physics Environment simulation
- [ ] Add Validation Agent for quality assurance
- [ ] Create Project Manager Agent for coordination
- [ ] Implement communication protocols
- [ ] Add data visualization capabilities

## 📚 Dependencies

- **LangChain** - AI agent framework
- **OpenAI** - Language model API
- **NumPy** - Scientific computing
- **Pydantic** - Data validation
- **Rich** - Terminal output formatting

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For questions or issues, please open a GitHub issue or contact me!
