import os
from dotenv import load_dotenv
from typing import TypedDict, Literal, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
import json
import requests
from enum import Enum

# Load environment variables from .env file
load_dotenv()

# Define the state structure
class RouterState(TypedDict):
    query: str
    intent: Optional[str]
    api_calls_needed: Optional[List[str]]
    current_api_call: Optional[str]
    extracted_params: Optional[Dict[str, Any]]
    api_responses: Optional[Dict[str, Any]]  # Changed to store multiple responses
    final_answer: Optional[str]
    error: Optional[str]
    needs_followup: Optional[bool]
    followup_params: Optional[Dict[str, Any]]

# Define API types
class APIType(str, Enum):
    EMPLOYEE = "employee_api"
    SEAL = "seal_api"
    UNKNOWN = "unknown"

# Define intent types
class IntentType(str, Enum):
    FIND_AO = "find_ao"  # Account Owner
    FIND_AO_WITH_EMAIL = "find_ao_with_email"  # Account Owner with email
    EMPLOYEE_INFO = "employee_info"
    PROJECT_INFO = "project_info"
    SEAL_INFO = "seal_info"
    UNKNOWN = "unknown"

# Pydantic models for structured output
class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent of the query")
    confidence: float = Field(description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Brief explanation of the classification")
    requires_multiple_calls: bool = Field(description="Whether this query requires multiple API calls")

class ParameterExtraction(BaseModel):
    parameters: Dict[str, Any] = Field(description="Extracted parameters from the query")
    api_calls_needed: List[str] = Field(description="List of APIs that need to be called in order")
    requires_followup: bool = Field(description="Whether a followup API call is needed")

# Mock API clients (replace with actual implementations)
class EmployeeAPIClient:
    def __init__(self, base_url: str = "https://api.company.com/employees"):
        self.base_url = base_url
    
    def get_employee_by_id(self, employee_id: str):
        # Mock response with email
        return {
            "employee_id": employee_id,
            "name": "John Doe",
            "role": "Account Owner",
            "department": "Sales",
            "email": "john.doe@company.com",
            "phone": "+1-555-0123"
        }
    
    def search_employees(self, query: Dict[str, Any]):
        # Mock response
        return {
            "employees": [
                {
                    "employee_id": "E123", 
                    "name": "John Doe", 
                    "role": "Account Owner",
                    "email": "john.doe@company.com"
                }
            ]
        }

class SealAPIClient:
    def __init__(self, base_url: str = "https://api.company.com/seals"):
        self.base_url = base_url
    
    def get_seal_by_id(self, seal_id: str):
        # Mock response
        return {
            "seal_id": seal_id,
            "project_name": "Project Alpha",
            "account_owner_id": "E123",  # This will be used for followup call
            "application_owner_id": "E123",  # Added application owner
            "status": "active"
        }
    
    def get_project_details(self, seal_id: str):
        # Mock response
        return {
            "seal_id": seal_id,
            "project_name": "Project Alpha",
            "account_owner": {
                "employee_id": "E123",
                "name": "John Doe"
            },
            "application_owner": {
                "employee_id": "E123", 
                "name": "John Doe"
            },
            "created_date": "2024-01-15",
            "status": "active"
        }

# Initialize clients
employee_client = EmployeeAPIClient()
seal_client = SealAPIClient()

# Helper function to make OpenAI API calls
def call_openai_api(system_prompt: str, user_prompt: str, response_format: Optional[Dict] = None) -> str:
    """Make a request to OpenAI API using requests.post"""
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1000
    }
    
    # Add response format for structured output if provided
    if response_format:
        payload["response_format"] = response_format
    
    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"OpenAI API request failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"Unexpected OpenAI API response format: {str(e)}")

# Helper function to parse JSON response with validation
def parse_structured_response(response_text: str, model_class: BaseModel) -> BaseModel:
    """Parse and validate JSON response using Pydantic model"""
    try:
        # Try to parse as JSON
        response_data = json.loads(response_text)
        return model_class(**response_data)
    except json.JSONDecodeError:
        # If not valid JSON, try to extract JSON from the text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_data = json.loads(json_match.group())
            return model_class(**response_data)
        else:
            raise Exception(f"Could not parse structured response: {response_text}")
    except Exception as e:
        raise Exception(f"Response validation failed: {str(e)}")

# Intent classifier node
def classify_intent(state: RouterState) -> RouterState:
    """Classify the intent of the user query"""
    
    system_prompt = """You are an intent classifier for an API routing system.
    Classify the user's query into one of these intents:
    - find_ao: Finding the Account Owner (AO) of a project or seal
    - find_ao_with_email: Finding the Account Owner with their email/contact details
    - employee_info: Getting information about an employee
    - project_info: Getting project details
    - seal_info: Getting seal information
    - unknown: Cannot determine the intent
    
    Pay special attention to queries that ask for additional information like email, phone, or contact details.
    These typically require multiple API calls and should be classified as requiring multiple calls.
    
    Examples:
    - "Who is the AO of seal 111811?" -> find_ao, single call
    - "Who is the application owner of seal 111811 and get his email?" -> find_ao_with_email, multiple calls
    - "Get the account owner's email for project 111811" -> find_ao_with_email, multiple calls
    
    Respond with a JSON object containing:
    - intent: The classified intent
    - confidence: Confidence score between 0 and 1
    - reasoning: Brief explanation of the classification
    - requires_multiple_calls: Whether this query requires multiple API calls (boolean)"""
    
    try:
        response_text = call_openai_api(system_prompt, state["query"])
        result = parse_structured_response(response_text, IntentClassification)
        
        state["intent"] = result.intent
        state["needs_followup"] = result.requires_multiple_calls
        return state
    except Exception as e:
        state["error"] = f"Intent classification failed: {str(e)}"
        state["intent"] = IntentType.UNKNOWN
        return state

# Parameter extractor node
def extract_parameters(state: RouterState) -> RouterState:
    """Extract parameters and determine which APIs to call"""
    
    system_prompt = """Extract parameters from the query and determine which APIs need to be called.
    
    APIs available:
    - employee_api: For employee-related queries
    - seal_api: For seal/project-related queries
    
    For queries that need multiple API calls, list them in the order they should be executed.
    For example:
    - "Who is the application owner of seal 111811 and get his email?" 
      -> First call seal_api to get owner ID, then employee_api to get email
    
    Extract relevant parameters like:
    - seal_id: Numeric ID for seal/project
    - employee_id: Employee identifier
    - employee_name: Name of employee
    - Other relevant fields
    
    Respond with a JSON object containing:
    - parameters: Dict of extracted parameters
    - api_calls_needed: List of APIs to call in order
    - requires_followup: Whether a followup API call is needed (boolean)"""
    
    user_prompt = f"Query: {state['query']}\nIntent: {state.get('intent', 'unknown')}"
    
    try:
        response_text = call_openai_api(system_prompt, user_prompt)
        result = parse_structured_response(response_text, ParameterExtraction)
        
        state["extracted_params"] = result.parameters
        state["api_calls_needed"] = result.api_calls_needed
        state["needs_followup"] = result.requires_followup
        
        # Set the first API call to execute
        if result.api_calls_needed:
            state["current_api_call"] = result.api_calls_needed[0]
        
        return state
    except Exception as e:
        state["error"] = f"Parameter extraction failed: {str(e)}"
        return state

# API router node
def route_to_api(state: RouterState) -> RouterState:
    """Route to the appropriate API based on intent and parameters"""
    
    try:
        current_api = state.get("current_api_call", APIType.UNKNOWN)
        params = state.get("extracted_params", {})
        
        # Initialize api_responses if not exists
        if not state.get("api_responses"):
            state["api_responses"] = {}
        
        if current_api == APIType.SEAL:
            # Handle Seal API calls
            if "seal_id" in params:
                if state["intent"] in [IntentType.FIND_AO_WITH_EMAIL, IntentType.FIND_AO]:
                    # Get seal details to find owner ID
                    seal_response = seal_client.get_seal_by_id(params["seal_id"])
                    state["api_responses"]["seal"] = seal_response
                    
                    # If we need followup, prepare parameters for employee API
                    if state.get("needs_followup"):
                        owner_id = seal_response.get("application_owner_id") or seal_response.get("account_owner_id")
                        if owner_id:
                            state["followup_params"] = {"employee_id": owner_id}
                else:
                    # Get full project details
                    project_response = seal_client.get_project_details(params["seal_id"])
                    state["api_responses"]["seal"] = project_response
            else:
                state["error"] = "Missing seal_id parameter"
                
        elif current_api == APIType.EMPLOYEE:
            # Handle Employee API calls
            # Use followup_params if available (from previous API call)
            employee_params = state.get("followup_params", params)
            
            if "employee_id" in employee_params:
                employee_response = employee_client.get_employee_by_id(employee_params["employee_id"])
                state["api_responses"]["employee"] = employee_response
            elif "employee_name" in employee_params:
                employee_response = employee_client.search_employees(employee_params)
                state["api_responses"]["employee"] = employee_response
            else:
                state["error"] = "Missing employee parameters"
        else:
            state["error"] = f"Unknown API type: {current_api}"
            
    except Exception as e:
        state["error"] = f"API call failed: {str(e)}"
    
    return state

# Response formatter node
def format_response(state: RouterState) -> RouterState:
    """Format the API response into a human-readable answer"""
    
    if state.get("error"):
        state["final_answer"] = f"I encountered an error: {state['error']}"
        return state
    
    system_prompt = """You are a helpful assistant that formats API responses into clear, 
    human-readable answers. Use the API response data to answer the original query concisely.
    
    When multiple API responses are available, combine the information appropriately.
    For example, if the user asked for an application owner's email, combine the seal 
    information with the employee information to provide a complete answer."""
    
    user_prompt = f"""Original query: {state['query']}
Intent: {state.get('intent', 'unknown')}
API Responses: {json.dumps(state.get('api_responses', {}), indent=2)}

Please provide a clear, complete answer to the user's question."""
    
    try:
        response_text = call_openai_api(system_prompt, user_prompt)
        state["final_answer"] = response_text
    except Exception as e:
        state["final_answer"] = f"Failed to format response: {str(e)}"
    
    return state

# Define routing logic
def should_continue(state: RouterState) -> Literal["extract_parameters", "route_to_api", "format_response", END]:
    """Determine next step based on current state"""
    
    if state.get("error"):
        return "format_response"
    
    if state.get("intent") == IntentType.UNKNOWN:
        return "format_response"
    
    if not state.get("extracted_params"):
        return "extract_parameters"
    
    # Check if we need to make more API calls
    api_calls_needed = state.get("api_calls_needed", [])
    api_responses = state.get("api_responses", {})
    
    if not api_calls_needed:
        return "format_response"
    
    # Check which API calls have been completed
    completed_calls = []
    if "seal" in api_responses:
        completed_calls.append(APIType.SEAL)
    if "employee" in api_responses:
        completed_calls.append(APIType.EMPLOYEE)
    
    # Find next API call to make
    for api_call in api_calls_needed:
        if api_call not in [call.value if hasattr(call, 'value') else call for call in completed_calls]:
            state["current_api_call"] = api_call
            return "route_to_api"
    
    # All API calls completed
    return "format_response"

# Build the graph
def create_router_graph():
    workflow = StateGraph(RouterState)
    
    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("extract_parameters", extract_parameters)
    workflow.add_node("route_to_api", route_to_api)
    workflow.add_node("format_response", format_response)
    
    # Set entry point
    workflow.set_entry_point("classify_intent")
    
    # Add edges
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "extract_parameters": "extract_parameters",
            "format_response": "format_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "extract_parameters",
        should_continue,
        {
            "route_to_api": "route_to_api",
            "format_response": "format_response",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "route_to_api",
        should_continue,
        {
            "route_to_api": "route_to_api",  # For multiple API calls
            "format_response": "format_response",
            END: END
        }
    )
    
    workflow.add_edge("format_response", END)
    
    return workflow.compile()

# Main router class
class IntentBasedAPIRouter:
    def __init__(self):
        self.graph = create_router_graph()
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route a query through the intent-based API router"""
        
        initial_state = RouterState(
            query=query,
            intent=None,
            api_calls_needed=None,
            current_api_call=None,
            extracted_params=None,
            api_responses=None,
            final_answer=None,
            error=None,
            needs_followup=None,
            followup_params=None
        )
        
        # Run the graph
        result = self.graph.invoke(initial_state)
        
        return {
            "query": result["query"],
            "intent": result.get("intent"),
            "api_calls_made": result.get("api_calls_needed"),
            "parameters": result.get("extracted_params"),
            "api_responses": result.get("api_responses"),
            "answer": result.get("final_answer"),
            "error": result.get("error")
        }

# Example usage
if __name__ == "__main__":
    # Initialize router
    router = IntentBasedAPIRouter()
    
    # Test queries including multi-step ones
    test_queries = [
        "Who is the AO of project with seal id 111811?",
        "Who is the application owner of seal 111811 and get his email?",
        "Find the account owner for seal 111811 and provide their contact information",
        "Get employee information for John Doe",
        "What are the project details for seal 111811?",
        "Show me seal information for ID 111811",
        "Get the email of the application owner for seal 111811"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        result = router.route_query(query)
        print(f"Intent: {result['intent']}")
        print(f"API Calls Made: {result['api_calls_made']}")
        print(f"Parameters: {result['parameters']}")
        print(f"API Responses: {json.dumps(result['api_responses'], indent=2) if result['api_responses'] else 'None'}")
        print(f"Answer: {result['answer']}")
        if result['error']:
            print(f"Error: {result['error']}")
        print("=" * 60)