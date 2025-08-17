#!/usr/bin/env python3
"""
Test script for CommunicationAgent functionality
"""

import sys
sys.path.append(sys.path[0])

from src.communication.protocol import CommunicationAgent, Message
from src.agents.core.experimental_agent import ExperimentalAgent
from src.agents.core.theoretical_agent import TheoreticalAgent


def test_communication_agent():
    """Test the CommunicationAgent functionality"""
    print("🧪 Testing Communication Agent...\n")
    
    # Initialize agents
    comm_agent = CommunicationAgent("CommAgent-1")
    exp_agent = ExperimentalAgent("ExpAgent-1")
    theo_agent = TheoreticalAgent("TheoAgent-1")
    
    # Register agents with communication system
    print("Registering agents...")
    comm_agent.register_agent("ExpAgent-1", ["experimental_design", "data_collection"], "active")
    comm_agent.register_agent("TheoAgent-1", ["hypothesis_generation", "mathematical_modeling"], "active")
    
    print("\nAgent Registry Status:")
    print(comm_agent.agent_executor.invoke({
    "input": "list registered agents",
    "chat_history": [],          
    "agent_scratchpad": []       
}))
    
    # Test message routing
    print("\nTesting message routing...")
    
    # Test 1: Valid message
    message1 = Message(
        sender="ExpAgent-1",
        recipient="TheoAgent-1",
        message_type="query",
        content="What theoretical framework should I use for pendulum analysis?",
        priority=3
    )
    result1 = comm_agent._route_message(message1)
    print(f"Message 1: {result1}")
    
    # Test 2: Broadcast message
    print("\nTesting broadcast...")
    broadcast_result = comm_agent.broadcast_message(
        "Starting pendulum experiment phase 1", 
        "notification"
    )
    print(f"Broadcast: {broadcast_result}")
    
    # Test 3: Get conversation history
    print("\nGetting conversation history...")
    history = comm_agent.get_conversation_history("ExpAgent-1", "TheoAgent-1")
    print(f"Conversation history: {len(history)} messages")
    for msg in history:
        print(f"  {msg.timestamp.strftime('%H:%M:%S')} - {msg.sender} → {msg.recipient}: {msg.content[:50]}...")
    
    # Test 4: Status report
    print("\nCommunication Agent Status Report:")
    print(comm_agent.get_status_report())
    
    # Test 5: LangChain integration
    print("\n🤖 Testing LangChain integration...")
    try:
        langchain_response = comm_agent.act({
            "input": "Send a message to ExpAgent-1 asking about experiment progress"
        })
        print(f"LangChain Response: {langchain_response}")
    except Exception as e:
        print(f"LangChain test failed (expected if no API key): {e}")
    
    print("\nCommunication Agent test completed!")


if __name__ == "__main__":
    test_communication_agent()
