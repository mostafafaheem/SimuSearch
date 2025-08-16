from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from .base_agent import BaseAgent


class Message(BaseModel):
    """Structured message for inter-agent communication"""
    sender: str
    recipient: str
    message_type: str = Field(description="Type of message: query, response, notification, command")
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=5, description="Priority: 1=low, 5=critical")


class CommunicationProtocol(BaseModel):
    """Protocol for agent communication"""
    protocol_name: str
    version: str = "1.0"
    message_formats: List[str] = Field(default_factory=list)
    routing_rules: Dict[str, List[str]] = Field(default_factory=dict)


class CommunicationAgent(BaseAgent):
    """
    Communication Agent responsible for:
    - Inter-agent message routing
    - Communication protocol management
    - Message formatting and validation
    - Coordination between research agents
    """
    
    def __init__(self, name: str, llm_model: str = "gpt-4", temperature: float = 0.1):
        super().__init__(name)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.message_queue: List[Message] = []
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.communication_protocols: Dict[str, CommunicationProtocol] = {}
        self.conversation_history: Dict[str, List[Message]] = {}
        
        # Initialize default communication protocol
        self._setup_default_protocol()
        
        # Setup LangChain tools
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _setup_default_protocol(self):
        """Setup default communication protocol"""
        default_protocol = CommunicationProtocol(
            protocol_name="SimuSearch_Research_v1",
            message_formats=["query", "response", "notification", "command", "data_share"],
            routing_rules={
                "experimental": ["analysis", "theoretical"],
                "theoretical": ["experimental", "analysis"],
                "analysis": ["experimental", "theoretical", "communication"],
                "communication": ["*"]  # Can communicate with all agents
            }
        )
        self.communication_protocols["default"] = default_protocol
    
    def _create_tools(self) -> List[BaseTool]:
        """Create LangChain tools for communication tasks"""
        from langchain.tools import tool
        
        @tool
        def send_message(recipient: str, content: str, message_type: str = "query", priority: int = 1) -> str:
            """Send a message to another agent"""
            message = Message(
                sender=self.name,
                recipient=recipient,
                message_type=message_type,
                content=content,
                priority=priority
            )
            return self._route_message(message)
        
        @tool
        def broadcast_message(content: str, message_type: str = "notification") -> str:
            """Broadcast a message to all registered agents"""
            recipients = list(self.agent_registry.keys())
            results = []
            for recipient in recipients:
                message = Message(
                    sender=self.name,
                    recipient=recipient,
                    message_type=message_type,
                    content=content
                )
                results.append(self._route_message(message))
            return f"Broadcasted to {len(recipients)} agents: {', '.join(results)}"
        
        @tool
        def get_agent_status(agent_name: str) -> str:
            """Get the current status of a specific agent"""
            if agent_name in self.agent_registry:
                agent_info = self.agent_registry[agent_name]
                return f"Agent {agent_name}: {agent_info.get('status', 'unknown')}"
            return f"Agent {agent_name} not found in registry"
        
        @tool
        def list_registered_agents() -> str:
            """List all registered agents and their capabilities"""
            if not self.agent_registry:
                return "No agents registered"
            
            agent_list = []
            for name, info in self.agent_registry.items():
                capabilities = info.get('capabilities', [])
                status = info.get('status', 'unknown')
                agent_list.append(f"- {name}: {status} (capabilities: {', '.join(capabilities)})")
            
            return "Registered Agents:\n" + "\n".join(agent_list)
        
        return [send_message, broadcast_message, get_agent_status, list_registered_agents]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create LangChain agent executor"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Communication Agent in a multi-agent scientific research simulator. 
            Your role is to facilitate communication between research agents, manage message routing, 
            and ensure effective coordination. Use the available tools to help with communication tasks.
            
            Always be clear, concise, and professional in your communications."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def register_agent(self, agent_name: str, capabilities: List[str], status: str = "active"):
        """Register a new agent in the communication system"""
        self.agent_registry[agent_name] = {
            "capabilities": capabilities,
            "status": status,
            "registered_at": datetime.now(),
            "last_seen": datetime.now()
        }
        print(f"✅ Registered agent: {agent_name} with capabilities: {capabilities}")
    
    def _route_message(self, message: Message) -> str:
        """Route a message to the appropriate recipient"""
        # Validate message format
        if not self._validate_message(message):
            return f"❌ Invalid message format: {message.message_type}"
        
        # Check routing rules
        if not self._check_routing_rules(message):
            return f"❌ Routing blocked: {message.sender} cannot send to {message.recipient}"
        
        # Add to conversation history
        conversation_key = f"{message.sender}_{message.recipient}"
        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []
        self.conversation_history[conversation_key].append(message)
        
        # Update agent last seen
        if message.sender in self.agent_registry:
            self.agent_registry[message.sender]["last_seen"] = datetime.now()
        
        return f"✅ Message routed: {message.sender} → {message.recipient} ({message.message_type})"
    
    def _validate_message(self, message: Message) -> bool:
        """Validate message format and content"""
        if not message.content or len(message.content.strip()) == 0:
            return False
        
        if message.message_type not in ["query", "response", "notification", "command", "data_share"]:
            return False
        
        if message.priority < 1 or message.priority > 5:
            return False
        
        return True
    
    def _check_routing_rules(self, message: Message) -> bool:
        """Check if message routing is allowed by protocol rules"""
        protocol = self.communication_protocols.get("default")
        if not protocol:
            return True  # Allow if no protocol defined
        
        sender_type = self._get_agent_type(message.sender)
        recipient_type = self._get_agent_type(message.recipient)
        
        if not sender_type or not recipient_type:
            return True  # Allow if agent types unknown
        
        allowed_recipients = protocol.routing_rules.get(sender_type, [])
        if "*" in allowed_recipients or recipient_type in allowed_recipients:
            return True
        
        return False
    
    def _get_agent_type(self, agent_name: str) -> Optional[str]:
        """Get the type/category of an agent"""
        if "experimental" in agent_name.lower():
            return "experimental"
        elif "theoretical" in agent_name.lower():
            return "theoretical"
        elif "analysis" in agent_name.lower():
            return "analysis"
        elif "communication" in agent_name.lower():
            return "communication"
        return None
    
    def get_conversation_history(self, agent1: str, agent2: str) -> List[Message]:
        """Get conversation history between two agents"""
        key1 = f"{agent1}_{agent2}"
        key2 = f"{agent2}_{agent1}"
        
        history = []
        if key1 in self.conversation_history:
            history.extend(self.conversation_history[key1])
        if key2 in self.conversation_history:
            history.extend(self.conversation_history[key2])
        
        # Sort by timestamp
        history.sort(key=lambda x: x.timestamp)
        return history
    
    def act(self, observations: Dict[str, Any]) -> str:
        """Main action method for the communication agent"""
        if "message" in observations:
            # Handle incoming message
            return self._handle_incoming_message(observations["message"])
        elif "command" in observations:
            # Handle command
            return self._handle_command(observations["command"])
        else:
            # Use LangChain agent for general communication tasks
            return self.agent_executor.invoke({"input": str(observations)})["output"]
    
    def _handle_incoming_message(self, message_data: Dict[str, Any]) -> str:
        """Handle incoming message from another agent"""
        try:
            message = Message(**message_data)
            return self._route_message(message)
        except Exception as e:
            return f"❌ Error processing message: {str(e)}"
    
    def _handle_command(self, command: str) -> str:
        """Handle direct commands"""
        if command == "status":
            return self.list_registered_agents()
        elif command == "clear_queue":
            self.message_queue.clear()
            return "✅ Message queue cleared"
        else:
            return f"❌ Unknown command: {command}"
    
    def get_status_report(self) -> str:
        """Generate a comprehensive status report"""
        report = f"📊 Communication Agent Status Report\n"
        report += f"Agent: {self.name}\n"
        report += f"Registered Agents: {len(self.agent_registry)}\n"
        report += f"Active Conversations: {len(self.conversation_history)}\n"
        report += f"Message Queue Size: {len(self.message_queue)}\n"
        report += f"Protocols: {len(self.communication_protocols)}\n"
        return report