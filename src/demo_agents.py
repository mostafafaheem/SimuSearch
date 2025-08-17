#!/usr/bin/env python3
"""
Simple demonstration of Experimental and Theoretical agents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.core.experimental_agent import ExperimentalAgent
from src.agents.core.theoretical_agent import TheoreticalAgent


def main():
    """Main demonstration function"""
    print("🚀 SimuSearch Multi-Agent Scientific Research Simulator")
    print("=" * 60)
    
    # Initialize agents
    print("\n📋 Initializing agents...")
    exp_agent = ExperimentalAgent("ExpAgent-1")
    theo_agent = TheoreticalAgent("TheoAgent-1")
    
    print(f"✅ {exp_agent.name} initialized")
    print(f"✅ {theo_agent.name} initialized")
    
    # Demonstrate agent capabilities
    print("\n🧮 Theoretical Agent Capabilities:")
    print("-" * 40)
    
    # Generate hypothesis
    print("1. Generating pendulum hypothesis...")
    hypothesis = theo_agent.generate_pendulum_hypothesis()
    print(f"   {hypothesis}")
    
    # Create mathematical model
    print("\n2. Creating mathematical model...")
    model = theo_agent.create_pendulum_model("simple")
    print(f"   {model}")
    
    # List models
    print("\n3. Available mathematical models:")
    print(theo_agent.list_models())
    
    print("\n🧪 Experimental Agent Capabilities:")
    print("-" * 40)
    
    # Check equipment
    print("1. Equipment status:")
    print(exp_agent.check_equipment_status())
    
    # Design experiment
    print("\n2. Designing pendulum experiment...")
    experiment = exp_agent.design_pendulum_experiment()
    print(f"   {experiment}")
    
    # List experiments
    print("\n3. Current experiments:")
    print(exp_agent.list_experiments())
    
    # Demonstrate collaboration
    print("\n🤝 Agent Collaboration:")
    print("-" * 40)
    
    print("Theoretical Agent provides framework → Experimental Agent tests it")
    print("Experimental Agent collects data → Theoretical Agent validates theory")
    
    # Show status reports
    print("\n📊 Final Status Reports:")
    print("-" * 40)
    
    print("Experimental Agent:")
    print(exp_agent.get_status_report())
    
    print("\nTheoretical Agent:")
    print(theo_agent.get_status_report())
    
    print("\n🎉 Demonstration completed!")
    print("\nNext steps:")
    print("1. Set up environment simulation")
    print("2. Implement communication protocols")
    print("3. Add analysis and validation agents")
    print("4. Run full pendulum experiment")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("This might be due to missing dependencies or API keys")
        print("Try running: pip install -r requirements.txt")
