#!/usr/bin/env python3
"""
Test script for Experimental and Theoretical agents working together
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.core.experimental_agent import ExperimentalAgent
from src.agents.core.theoretical_agent import TheoreticalAgent


def test_agents_collaboration():
    """Test the collaboration between Experimental and Theoretical agents"""
    print("🔬 Testing Experimental and Theoretical Agent Collaboration...\n")
    
    # Initialize agents
    exp_agent = ExperimentalAgent("ExpAgent-1")
    theo_agent = TheoreticalAgent("TheoAgent-1")
    
    print("📝 Step 1: Theoretical Agent generates hypothesis...")
    hypothesis_result = theo_agent.generate_pendulum_hypothesis()
    print(f"Result: {hypothesis_result}")
    
    print("\n📐 Step 2: Theoretical Agent creates mathematical model...")
    model_result = theo_agent.create_pendulum_model("simple")
    print(f"Result: {model_result}")
    
    print("\n🧪 Step 3: Experimental Agent designs experiment...")
    experiment_result = exp_agent.design_pendulum_experiment()
    print(f"Result: {experiment_result}")
    
    print("\n📊 Step 4: Check agent statuses...")
    print("\nExperimental Agent Status:")
    print(exp_agent.get_status_report())
    
    print("\nTheoretical Agent Status:")
    print(theo_agent.get_status_report())
    
    print("\n📋 Step 5: List experiments and hypotheses...")
    print("\nExperiments:")
    print(exp_agent.list_experiments())
    
    print("\nHypotheses:")
    print(theo_agent.list_hypotheses())
    
    print("\n🔍 Step 6: Search literature...")
    literature_result = theo_agent.search_literature("pendulum")
    print(f"Literature Search: {literature_result}")
    
    print("\n✅ Agent collaboration test completed!")


def test_individual_agent_capabilities():
    """Test individual agent capabilities"""
    print("\n🧪 Testing Individual Agent Capabilities...\n")
    
    # Test Experimental Agent
    exp_agent = ExperimentalAgent("ExpAgent-Test")
    
    print("📝 Testing Experimental Agent tools...")
    print("1. Equipment status:")
    print(exp_agent.check_equipment_status())
    
    print("\n2. Design experiment:")
    variables = {"independent": ["length"], "dependent": ["period"], "controlled": ["mass"]}
    parameters = {"lengths": [0.1, 0.2, 0.3], "trials": 3}
    result = exp_agent.design_experiment("Test hypothesis", variables, parameters)
    print(f"Result: {result}")
    
    # Test Theoretical Agent
    theo_agent = TheoreticalAgent("TheoAgent-Test")
    
    print("\n🧮 Testing Theoretical Agent tools...")
    print("1. List models:")
    print(theo_agent.list_models())
    
    print("\n2. Add literature reference:")
    authors = ["Einstein, A.", "Infeld, L."]
    findings = ["Relativity theory", "Space-time curvature"]
    lit_result = theo_agent.add_literature_reference(
        "The Evolution of Physics", authors, 1938, "Cambridge", findings
    )
    print(f"Result: {lit_result}")
    
    print("\n3. Search literature:")
    search_result = theo_agent.search_literature("physics")
    print(f"Search: {search_result}")


def test_pendulum_experiment_workflow():
    """Test a complete pendulum experiment workflow"""
    print("\n🎯 Testing Complete Pendulum Experiment Workflow...\n")
    
    # Initialize agents
    exp_agent = ExperimentalAgent("ExpAgent-Workflow")
    theo_agent = TheoreticalAgent("TheoAgent-Workflow")
    
    print("📋 Phase 1: Theory Development")
    print("1. Generate hypothesis...")
    hypothesis = theo_agent.generate_pendulum_hypothesis()
    print(f"   {hypothesis}")
    
    print("2. Create mathematical model...")
    model = theo_agent.create_pendulum_model("simple")
    print(f"   {model}")
    
    print("\n🧪 Phase 2: Experiment Design")
    print("1. Design experiment...")
    experiment = exp_agent.design_pendulum_experiment()
    print(f"   {experiment}")
    
    print("2. Start experiment...")
    start_result = exp_agent.start_experiment("exp_001")
    print(f"   {start_result}")
    
    print("\n📊 Phase 3: Data Collection")
    print("1. Collect data points...")
    data_results = []
    for i, length in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        # Simulate pendulum period data (T = 2π√(L/g))
        import math
        period = 2 * math.pi * math.sqrt(length / 9.81)
        data_result = exp_agent.collect_data("exp_001", "length", length)
        data_results.append(data_result)
        data_result = exp_agent.collect_data("exp_001", "period", period, 0.01)
        data_results.append(data_result)
    
    print("   Data collection completed")
    
    print("\n✅ Phase 4: Experiment Completion")
    print("1. Complete experiment...")
    complete_result = exp_agent.complete_experiment("exp_001")
    print(f"   {complete_result}")
    
    print("\n📈 Final Results:")
    print("Experimental Agent Status:")
    print(exp_agent.get_status_report())
    
    print("\nTheoretical Agent Status:")
    print(theo_agent.get_status_report())


if __name__ == "__main__":
    try:
        test_agents_collaboration()
        test_individual_agent_capabilities()
        test_pendulum_experiment_workflow()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
