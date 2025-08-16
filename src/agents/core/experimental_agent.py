from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import numpy as np
from dataclasses import dataclass
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from ..base_agent import BaseAgent


@dataclass
class Experiment:
    """Represents a scientific experiment"""
    id: str
    name: str
    hypothesis: str
    variables: Dict[str, Any]
    parameters: Dict[str, Any]
    status: str = "planned"  # planned, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


@dataclass
class DataPoint:
    """Represents a single data measurement"""
    timestamp: datetime
    variable: str
    value: float
    uncertainty: float
    metadata: Dict[str, Any] = None


class ExperimentalAgent(BaseAgent):
    """
    Experimental Agent responsible for:
    - Experimental design and planning
    - Data collection and measurement
    - Hypothesis testing
    - Equipment management
    - Quality control and validation
    """
    
    def __init__(self, name: str, llm_model: str = "gpt-4", temperature: float = 0.1):
        super().__init__(name)
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.experiments: Dict[str, Experiment] = {}
        self.data_collection: Dict[str, List[DataPoint]] = {}
        self.equipment_status: Dict[str, str] = {}
        self.quality_metrics: Dict[str, float] = {}
        self.experimental_protocols: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default equipment
        self._setup_default_equipment()
        
        # Setup LangChain tools
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent_executor()
    
    def _setup_default_equipment(self):
        """Setup default experimental equipment"""
        self.equipment_status = {
            "pendulum_setup": "available",
            "data_logger": "available", 
            "stopwatch": "available",
            "ruler": "available",
            "protractor": "available"
        }
    
    def _create_tools(self) -> List[BaseTool]:
        """Create LangChain tools for experimental tasks"""
        from langchain.tools import tool
        
        @tool
        def design_experiment(hypothesis: str, variables: str, parameters: str) -> str:
            """Design a new experiment based on hypothesis and requirements"""
            try:
                variables_dict = json.loads(variables) if isinstance(variables, str) else variables
                parameters_dict = json.loads(parameters) if isinstance(parameters, str) else parameters
                
                experiment_id = f"exp_{len(self.experiments) + 1:03d}"
                experiment = Experiment(
                    id=experiment_id,
                    name=f"Experiment {experiment_id}",
                    hypothesis=hypothesis,
                    variables=variables_dict,
                    parameters=parameters_dict,
                    metadata={"designed_by": self.name, "design_time": datetime.now()}
                )
                
                self.experiments[experiment_id] = experiment
                return f"✅ Experiment {experiment_id} designed successfully: {hypothesis}"
            except Exception as e:
                return f"❌ Failed to design experiment: {str(e)}"
        
        @tool
        def start_experiment(experiment_id: str) -> str:
            """Start a planned experiment"""
            if experiment_id not in self.experiments:
                return f"❌ Experiment {experiment_id} not found"
            
            experiment = self.experiments[experiment_id]
            if experiment.status != "planned":
                return f"❌ Experiment {experiment_id} cannot be started (status: {experiment.status})"
            
            # Check equipment availability
            required_equipment = experiment.parameters.get("required_equipment", [])
            unavailable_equipment = [eq for eq in required_equipment if self.equipment_status.get(eq) != "available"]
            
            if unavailable_equipment:
                return f"❌ Equipment unavailable: {unavailable_equipment}"
            
            # Start experiment
            experiment.status = "running"
            experiment.start_time = datetime.now()
            
            # Initialize data collection
            self.data_collection[experiment_id] = []
            
            return f"✅ Experiment {experiment_id} started at {experiment.start_time}"
        
        @tool
        def collect_data(experiment_id: str, variable: str, value: float, uncertainty: float = 0.0) -> str:
            """Collect a data point during an experiment"""
            if experiment_id not in self.experiments:
                return f"❌ Experiment {experiment_id} not found"
            
            experiment = self.experiments[experiment_id]
            if experiment.status != "running":
                return f"❌ Experiment {experiment_id} is not running (status: {experiment.status})"
            
            # Create data point
            data_point = DataPoint(
                timestamp=datetime.now(),
                variable=variable,
                value=value,
                uncertainty=uncertainty,
                metadata={"collected_by": self.name}
            )
            
            # Store data
            if experiment_id not in self.data_collection:
                self.data_collection[experiment_id] = []
            self.data_collection[experiment_id].append(data_point)
            
            return f"✅ Data collected: {variable} = {value} ± {uncertainty}"
        
        @tool
        def complete_experiment(experiment_id: str) -> str:
            """Complete a running experiment"""
            if experiment_id not in self.experiments:
                return f"❌ Experiment {experiment_id} not found"
            
            experiment = self.experiments[experiment_id]
            if experiment.status != "running":
                return f"❌ Experiment {experiment_id} is not running (status: {experiment.status})"
            
            # Complete experiment
            experiment.status = "completed"
            experiment.end_time = datetime.now()
            
            # Calculate basic statistics
            if experiment_id in self.data_collection:
                data = self.data_collection[experiment_id]
                experiment.results = self._calculate_statistics(data)
            
            return f"✅ Experiment {experiment_id} completed. Results: {experiment.results}"
        
        @tool
        def get_experiment_status(experiment_id: str) -> str:
            """Get the current status of an experiment"""
            if experiment_id not in self.experiments:
                return f"❌ Experiment {experiment_id} not found"
            
            experiment = self.experiments[experiment_id]
            status_info = f"Experiment {experiment_id}: {experiment.status}"
            
            if experiment.start_time:
                status_info += f" (Started: {experiment.start_time.strftime('%H:%M:%S')})"
            if experiment.end_time:
                status_info += f" (Completed: {experiment.end_time.strftime('%H:%M:%S')})"
            if experiment.results:
                status_info += f" (Results: {experiment.results})"
            
            return status_info
        
        @tool
        def list_experiments() -> str:
            """List all experiments and their status"""
            if not self.experiments:
                return "No experiments found"
            
            experiment_list = []
            for exp_id, experiment in self.experiments.items():
                status = f"{exp_id}: {experiment.status}"
                if experiment.hypothesis:
                    status += f" - {experiment.hypothesis[:50]}..."
                experiment_list.append(status)
            
            return "Experiments:\n" + "\n".join(experiment_list)
        
        @tool
        def check_equipment_status() -> str:
            """Check the status of all experimental equipment"""
            status_list = []
            for equipment, status in self.equipment_status.items():
                status_list.append(f"{equipment}: {status}")
            
            return "Equipment Status:\n" + "\n".join(status_list)
        
        return [design_experiment, start_experiment, collect_data, complete_experiment, 
                get_experiment_status, list_experiments, check_equipment_status]
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create LangChain agent executor"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Experimental Agent in a multi-agent scientific research simulator. 
            Your role is to design, conduct, and manage scientific experiments. You excel at:
            - Experimental design and methodology
            - Data collection and measurement
            - Quality control and validation
            - Equipment management
            - Hypothesis testing
            
            Use the available tools to help with experimental tasks. Always maintain scientific rigor
            and document your procedures carefully."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)
    
    def _calculate_statistics(self, data: List[DataPoint]) -> Dict[str, Any]:
        """Calculate basic statistics for collected data"""
        if not data:
            return {}
        
        # Group data by variable
        variable_data = {}
        for point in data:
            if point.variable not in variable_data:
                variable_data[point.variable] = []
            variable_data[point.variable].append(point.value)
        
        # Calculate statistics for each variable
        stats = {}
        for variable, values in variable_data.items():
            values_array = np.array(values)
            stats[variable] = {
                "count": len(values),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
                "range": float(np.max(values_array) - np.min(values_array))
            }
        
        return stats
    
    def design_pendulum_experiment(self, hypothesis: str = "Pendulum period depends on length") -> str:
        """Design a specific pendulum experiment"""
        variables = {
            "independent": ["length"],
            "dependent": ["period"],
            "controlled": ["mass", "amplitude", "environment"]
        }
        
        parameters = {
            "lengths": [0.1, 0.2, 0.3, 0.4, 0.5],  # meters
            "trials_per_length": 5,
            "measurement_method": "stopwatch",
            "required_equipment": ["pendulum_setup", "stopwatch", "ruler"],
            "data_points": 25  # 5 lengths × 5 trials
        }
        
        return self.design_experiment(hypothesis, variables, parameters)
    
    def act(self, observations: Dict[str, Any]) -> str:
        """Main action method for the experimental agent"""
        if "command" in observations:
            command = observations["command"]
            if command == "design_pendulum":
                return self.design_pendulum_experiment()
            elif command == "status":
                return self.list_experiments()
            elif command == "equipment":
                return self.check_equipment_status()
        
        # Use LangChain agent for general experimental tasks
        try:
            return self.agent_executor.invoke({"input": str(observations)})["output"]
        except Exception as e:
            return f"❌ Experimental agent error: {str(e)}"
    
    def get_status_report(self) -> str:
        """Generate a comprehensive status report"""
        report = f"🧪 Experimental Agent Status Report\n"
        report += f"Agent: {self.name}\n"
        report += f"Total Experiments: {len(self.experiments)}\n"
        report += f"Running Experiments: {sum(1 for exp in self.experiments.values() if exp.status == 'running')}\n"
        report += f"Completed Experiments: {sum(1 for exp in self.experiments.values() if exp.status == 'completed')}\n"
        report += f"Available Equipment: {sum(1 for status in self.equipment_status.values() if status == 'available')}\n"
        return report
