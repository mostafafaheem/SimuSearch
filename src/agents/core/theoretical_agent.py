from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import math
import numpy as np
from dataclasses import dataclass
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from ..base_agent import BaseAgent
from ...config import api_key


@dataclass
class Hypothesis:
    """Represents a scientific hypothesis"""
    id: str
    created_at: datetime
    statement: str
    confidence: float  # 0.0 to 1.0
    evidence: List[str]
    status: str = "proposed"  # proposed, testing, supported, refuted
    predictions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = None


@dataclass
class MathematicalModel:
    """Represents a mathematical model"""
    id: str
    created_at: datetime
    name: str
    equations: List[str]
    parameters: Dict[str, float]
    assumptions: List[str]
    domain_validity: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class LiteratureReference:
    """Represents a literature reference"""
    id: str
    title: str
    authors: List[str]
    year: int
    journal: str
    relevance_score: float  # 0.0 to 1.0
    key_findings: List[str]
    added_at: datetime
    doi: Optional[str] = None


class TheoreticalAgent(BaseAgent):
    """
    Theoretical Agent responsible for:
    - Hypothesis generation and evaluation
    - Mathematical modeling and equations
    - Literature review and synthesis
    - Theoretical framework development
    - Prediction generation and validation
    """
    
    def __init__(self, name: str, llm_model: str = "gemini-pro", temperature: float = 0.3):
        super().__init__(name)
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=api_key,
            temperature=temperature
        )
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.mathematical_models: Dict[str, MathematicalModel] = {}
        self.literature_database: Dict[str, LiteratureReference] = {}
        self.theoretical_frameworks: Dict[str, Dict[str, Any]] = {}
        self.prediction_models: Dict[str, Dict[str, Any]] = {}
        
        # Initialize with basic physics knowledge
        self._setup_basic_physics_models()
        
        # Setup LangChain tools
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _setup_basic_physics_models(self):
        """Setup basic physics models for common phenomena"""
        # Simple pendulum model
        pendulum_model = MathematicalModel(
            id="pendulum_simple",
            name="Simple Pendulum Period",
            equations=["T = 2π√(L/g)"],
            parameters={"g": 9.81, "π": math.pi},
            assumptions=["small amplitude", "point mass", "frictionless"],
            domain_validity={"amplitude": "< 15°", "length": "> 0.1m"},
            created_at=datetime.now(),
            metadata={"category": "mechanics", "complexity": "basic"}
        )
        self.mathematical_models["pendulum_simple"] = pendulum_model
        
        # Add basic literature
        basic_lit = LiteratureReference(
            id="galileo_1638",
            title="Discourses and Mathematical Demonstrations Relating to Two New Sciences",
            authors=["Galileo Galilei"],
            year=1638,
            journal="Leiden",
            relevance_score=0.9,
            key_findings=["Pendulum period independence from mass", "Isochronism of pendulum"],
            added_at=datetime.now()
        )
        self.literature_database["galileo_1638"] = basic_lit
    
    def _create_tools(self) -> List[BaseTool]:
        """Create LangChain tools for theoretical tasks"""
        from langchain.tools import tool
        
        @tool
        def generate_hypothesis(topic: str, context: str = "") -> str:
            """Generate a new scientific hypothesis based on topic and context"""
            try:
                hypothesis_id = f"hyp_{len(self.hypotheses) + 1:03d}"
                
                # Use LLM to generate hypothesis
                prompt = f"Generate a testable scientific hypothesis about {topic}. Context: {context}"
                response = self.llm.invoke(prompt)
                hypothesis_statement = response.content
                
                hypothesis = Hypothesis(
                    id=hypothesis_id,
                    statement=hypothesis_statement,
                    confidence=0.7,  # Initial confidence
                    evidence=[],
                    predictions=[],
                    created_at=datetime.now(),
                    metadata={"topic": topic, "context": context, "generated_by": self.name}
                )
                
                self.hypotheses[hypothesis_id] = hypothesis
                return f"✅ Hypothesis {hypothesis_id} generated: {hypothesis_statement[:100]}..."
            except Exception as e:
                return f"❌ Failed to generate hypothesis: {str(e)}"
        
        @tool
        def create_mathematical_model(name: str, equations: str, parameters: str) -> str:
            """Create a new mathematical model with equations and parameters"""
            try:
                equations_list = json.loads(equations) if isinstance(equations, str) else equations
                parameters_dict = json.loads(parameters) if isinstance(parameters, str) else parameters
                
                model_id = f"model_{len(self.mathematical_models) + 1:03d}"
                
                model = MathematicalModel(
                    id=model_id,
                    name=name,
                    equations=equations_list,
                    parameters=parameters_dict,
                    assumptions=[],
                    domain_validity={},
                    created_at=datetime.now(),
                    metadata={"created_by": self.name}
                )
                
                self.mathematical_models[model_id] = model
                return f"✅ Mathematical model {model_id} created: {name}"
            except Exception as e:
                return f"❌ Failed to create model: {str(e)}"
        
        @tool
        def add_literature_reference(title: str, authors: str, year: int, journal: str, key_findings: str) -> str:
            """Add a new literature reference to the database"""
            try:
                authors_list = json.loads(authors) if isinstance(authors, str) else authors
                findings_list = json.loads(key_findings) if isinstance(key_findings, str) else key_findings
                
                ref_id = f"ref_{len(self.literature_database) + 1:03d}"
                
                reference = LiteratureReference(
                    id=ref_id,
                    title=title,
                    authors=authors_list,
                    year=year,
                    journal=journal,
                    relevance_score=0.8,  # Default relevance
                    key_findings=findings_list,
                    added_at=datetime.now()
                )
                
                self.literature_database[ref_id] = reference
                return f"✅ Literature reference {ref_id} added: {title[:50]}..."
            except Exception as e:
                return f"❌ Failed to add reference: {str(e)}"
        
        @tool
        def search_literature(query: str) -> str:
            """Search literature database for relevant references"""
            if not self.literature_database:
                return "No literature references in database"
            
            relevant_refs = []
            query_lower = query.lower()
            
            for ref_id, reference in self.literature_database.items():
                # Simple keyword matching
                if (query_lower in reference.title.lower() or 
                    any(query_lower in finding.lower() for finding in reference.key_findings) or
                    any(query_lower in author.lower() for author in reference.authors)):
                    relevant_refs.append(f"{ref_id}: {reference.title} ({reference.year})")
            
            if not relevant_refs:
                return f"No literature found matching: {query}"
            
            return f"Literature matching '{query}':\n" + "\n".join(relevant_refs[:5])  # Limit to 5 results
        
        @tool
        def evaluate_hypothesis(hypothesis_id: str, new_evidence: str) -> str:
            """Evaluate a hypothesis based on new evidence"""
            if hypothesis_id not in self.hypotheses:
                return f"❌ Hypothesis {hypothesis_id} not found"
            
            hypothesis = self.hypotheses[hypothesis_id]
            
            # Add new evidence
            hypothesis.evidence.append(new_evidence)
            
            # Update confidence based on evidence (simple heuristic)
            if "support" in new_evidence.lower() or "confirm" in new_evidence.lower():
                hypothesis.confidence = min(1.0, hypothesis.confidence + 0.1)
            elif "contradict" in new_evidence.lower() or "refute" in new_evidence.lower():
                hypothesis.confidence = max(0.0, hypothesis.confidence - 0.2)
            
            # Update status
            if hypothesis.confidence > 0.8:
                hypothesis.status = "supported"
            elif hypothesis.confidence < 0.3:
                hypothesis.status = "refuted"
            else:
                hypothesis.status = "testing"
            
            return f"✅ Hypothesis {hypothesis_id} evaluated. New confidence: {hypothesis.confidence:.2f}, Status: {hypothesis.status}"
        
        @tool
        def generate_predictions(hypothesis_id: str) -> str:
            """Generate testable predictions from a hypothesis"""
            if hypothesis_id not in self.hypotheses:
                return f"❌ Hypothesis {hypothesis_id} not found"
            
            hypothesis = self.hypotheses[hypothesis_id]
            
            # Use LLM to generate predictions
            prompt = f"Generate 3 testable predictions from this hypothesis: {hypothesis.statement}"
            response = self.llm.invoke(prompt)
            
            # Parse predictions (simple approach)
            predictions = [pred.strip() for pred in response.content.split('\n') if pred.strip()][:3]
            hypothesis.predictions.extend(predictions)
            
            return f"✅ Generated {len(predictions)} predictions for hypothesis {hypothesis_id}:\n" + "\n".join(predictions)
        
        @tool
        def list_hypotheses() -> str:
            """List all hypotheses and their status"""
            if not self.hypotheses:
                return "No hypotheses found"
            
            hypothesis_list = []
            for hyp_id, hypothesis in self.hypotheses.items():
                status = f"{hyp_id}: {hypothesis.status} (confidence: {hypothesis.confidence:.2f})"
                if hypothesis.statement:
                    status += f" - {hypothesis.statement[:60]}..."
                hypothesis_list.append(status)
            
            return "Hypotheses:\n" + "\n".join(hypothesis_list)
        
        @tool
        def list_models() -> str:
            """List all mathematical models"""
            if not self.mathematical_models:
                return "No mathematical models found"
            
            model_list = []
            for model_id, model in self.mathematical_models.items():
                status = f"{model_id}: {model.name}"
                if model.equations:
                    status += f" - {model.equations[0][:50]}..."
                model_list.append(status)
            
            return "Mathematical Models:\n" + "\n".join(model_list)
        
        return [generate_hypothesis, create_mathematical_model, add_literature_reference, 
                search_literature, evaluate_hypothesis, generate_predictions, 
                list_hypotheses, list_models]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create LangChain agent executor"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Theoretical Agent in a multi-agent scientific research simulator. 
            Your role is to develop theoretical frameworks, generate hypotheses, and create mathematical models. 
            You excel at:
            - Hypothesis generation and evaluation
            - Mathematical modeling and equations
            - Literature review and synthesis
            - Theoretical framework development
            - Prediction generation and validation
            
            Use the available tools to help with theoretical tasks. Always maintain scientific rigor
            and provide well-reasoned theoretical insights."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def create_pendulum_model(self, complexity: str = "simple") -> str:
        """Create a mathematical model for pendulum motion"""
        if complexity == "simple":
            equations = ["T = 2π√(L/g)"]
            parameters = {"g": 9.81, "π": math.pi}
            assumptions = ["small amplitude", "point mass", "frictionless", "rigid rod"]
        elif complexity == "advanced":
            equations = [
                "T = 2π√(L/g) * (1 + (θ²/16) + (11θ⁴/3072) + ...)",
                "θ(t) = θ₀ * cos(ωt + φ)",
                "ω = √(g/L)"
            ]
            parameters = {"g": 9.81, "π": math.pi}
            assumptions = ["finite amplitude", "point mass", "frictionless", "rigid rod"]
        else:
            return "❌ Invalid complexity level. Use 'simple' or 'advanced'"
        
        return self.create_mathematical_model(
            f"Pendulum Model ({complexity})",
            equations,
            parameters
        )
    
    def generate_pendulum_hypothesis(self) -> str:
        """Generate a hypothesis about pendulum behavior"""
        return self.generate_hypothesis(
            "pendulum motion",
            "Investigating the relationship between pendulum length and period"
        )
    
    def act(self, observations: Dict[str, Any]) -> str:
        """Main action method for the theoretical agent"""
        if "command" in observations:
            command = observations["command"]
            if command == "create_pendulum_model":
                complexity = observations.get("complexity", "simple")
                return self.create_pendulum_model(complexity)
            elif command == "generate_pendulum_hypothesis":
                return self.generate_pendulum_hypothesis()
            elif command == "list_hypotheses":
                return self.list_hypotheses()
            elif command == "list_models":
                return self.list_models()
        
        # Use LangChain agent for general theoretical tasks
        try:
            return self.agent_executor.invoke({"input": str(observations)})["output"]
        except Exception as e:
            return f"❌ Theoretical agent error: {str(e)}"
    
    def get_status_report(self) -> str:
        """Generate a comprehensive status report"""
        report = f"🧮 Theoretical Agent Status Report\n"
        report += f"Agent: {self.name}\n"
        report += f"Total Hypotheses: {len(self.hypotheses)}\n"
        report += f"Mathematical Models: {len(self.mathematical_models)}\n"
        report += f"Literature References: {len(self.literature_database)}\n"
        report += f"Active Hypotheses: {sum(1 for hyp in self.hypotheses.values() if hyp.status == 'testing')}\n"
        return report
