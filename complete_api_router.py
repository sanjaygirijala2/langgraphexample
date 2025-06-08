#!/usr/bin/env python3
"""
Complete API Router with LangGraph
Ready-to-run Python file with mock data and real HTTP support

Installation:
pip install langgraph langchain-core requests python-dotenv

Usage:
python complete_api_router.py

Features:
- LangGraph workflow orchestration
- Mock enterprise APIs (Employee, Projects, Leave, Calendar)
- Real HTTP API support
- Swagger/OpenAPI integration
- Natural language query processing
- Easy to extend with new APIs

Author: API Whisperers Team
Version: 1.0
"""

import os
import json
import requests
import re
import time
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum
from dataclasses import dataclass

# Check if LangGraph is available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    print("⚠️ LangGraph not installed. Install with: pip install langgraph langchain-core")
    LANGGRAPH_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

# =============================================================================
# STATE DEFINITIONS FOR LANGGRAPH
# =============================================================================

class RouterState(TypedDict):
    """State that flows through the LangGraph workflow"""
    original_question: str
    intent: str
    parameters: Dict[str, Any]
    apis_to_call: List[str]
    api_responses: Dict[str, Any]
    final_answer: str
    error: Optional[str]

class QueryIntent(str, Enum):
    """Common query intents for enterprise scenarios"""
    GET_EMPLOYEE = "get_employee"
    GET_PROJECT_OWNER = "get_project_owner"
    CHECK_VACATION_STATUS = "check_vacation_status"
    GET_PROJECT_TEAM = "get_project_team"
    CHECK_AVAILABILITY = "check_availability"
    GET_PERFORMANCE_REVIEW = "get_performance_review"
    COMPLEX_QUERY = "complex_query"
    UNKNOWN = "unknown"

# =============================================================================
# MOCK DATA - REALISTIC ENTERPRISE DATA
# =============================================================================

MOCK_EMPLOYEES = {
    "E123": {
        "employee_id": "E123",
        "name": "John Doe",
        "email": "john.doe@company.com",
        "department": "Engineering",
        "role": "Senior Developer",
        "manager_id": "M456",
        "hire_date": "2020-03-15",
        "phone": "+1-555-0123",
        "status": "active"
    },
    "E124": {
        "employee_id": "E124",
        "name": "Sarah Johnson",
        "email": "sarah.johnson@company.com",
        "department": "Sales",
        "role": "Account Manager",
        "manager_id": "M457",
        "hire_date": "2021-07-20",
        "phone": "+1-555-0124",
        "status": "active"
    },
    "M456": {
        "employee_id": "M456",
        "name": "Jane Smith",
        "email": "jane.smith@company.com",
        "department": "Engineering",
        "role": "Engineering Manager",
        "manager_id": "VP001",
        "hire_date": "2018-01-10",
        "phone": "+1-555-0456",
        "status": "active"
    },
    "M457": {
        "employee_id": "M457",
        "name": "Mike Wilson",
        "email": "mike.wilson@company.com",
        "department": "Sales",
        "role": "Sales Manager",
        "manager_id": "VP002",
        "hire_date": "2019-05-15",
        "phone": "+1-555-0457",
        "status": "active"
    }
}

MOCK_SEALS = {
    "111811": {
        "seal_id": "111811",
        "project_name": "Project Alpha",
        "account_owner_id": "E123",
        "account_owner_name": "John Doe",
        "status": "active",
        "budget": "$150,000",
        "start_date": "2024-01-15",
        "end_date": "2024-12-31",
        "team_members": ["E123", "E124"],
        "completion": "65%"
    },
    "222822": {
        "seal_id": "222822",
        "project_name": "Project Beta",
        "account_owner_id": "E124",
        "account_owner_name": "Sarah Johnson",
        "status": "active",
        "budget": "$200,000",
        "start_date": "2024-03-01",
        "end_date": "2025-02-28",
        "team_members": ["E124", "M456"],
        "completion": "45%"
    }
}

MOCK_LEAVES = {
    "E123": {
        "employee_id": "E123",
        "current_leave": {
            "status": "on_leave",
            "type": "Annual",
            "start_date": "2024-12-20",
            "end_date": "2024-12-30",
            "reason": "Holiday vacation"
        },
        "leave_balance": {"annual": 15, "sick": 10, "personal": 5}
    },
    "E124": {
        "employee_id": "E124",
        "current_leave": {
            "status": "working",
            "type": None,
            "start_date": None,
            "end_date": None
        },
        "leave_balance": {"annual": 20, "sick": 12, "personal": 3}
    }
}

MOCK_CALENDAR = {
    "E123": {
        "employee_id": "E123",
        "today_meetings": [
            {
                "time": "09:00-10:00",
                "title": "Daily Standup",
                "attendees": ["E123", "E124", "M456"]
            },
            {
                "time": "14:00-15:30",
                "title": "Project Alpha Review",
                "attendees": ["E123", "M456"]
            }
        ]
    },
    "E124": {
        "employee_id": "E124",
        "today_meetings": [
            {
                "time": "09:00-10:00",
                "title": "Daily Standup",
                "attendees": ["E123", "E124", "M456"]
            },
            {
                "time": "11:00-12:00",
                "title": "Sales Pipeline Review",
                "attendees": ["E124", "M457"]
            }
        ]
    }
}

# New: Performance Review Data
MOCK_PERFORMANCE = {
    "E123": {
        "employee_id": "E123",
        "latest_review": {
            "date": "2024-06-15",
            "rating": "Exceeds Expectations",
            "score": 4.2,
            "goals_met": 8,
            "goals_total": 10,
            "reviewer": "Jane Smith"
        },
        "next_review": "2024-12-15"
    },
    "E124": {
        "employee_id": "E124",
        "latest_review": {
            "date": "2024-07-20",
            "rating": "Meets Expectations",
            "score": 3.8,
            "goals_met": 7,
            "goals_total": 9,
            "reviewer": "Mike Wilson"
        },
        "next_review": "2025-01-20"
    }
}

# =============================================================================
# HTTP CLIENT AND LLM FUNCTIONS
# =============================================================================

def make_api_call(url: str, method: str = "GET", params: Dict = None, headers: Dict = None) -> Dict[str, Any]:
    """Make HTTP request to API or return mock data"""
    
    # Handle mock APIs
    if "mock-employee-api" in url:
        return mock_employee_api(url.split("/")[-1], params)
    elif "mock-seal-api" in url:
        return mock_seal_api(url.split("/")[-1], params)
    elif "mock-leave-api" in url:
        return mock_leave_api(url.split("/")[-1], params)
    elif "mock-calendar-api" in url:
        return mock_calendar_api(url.split("/")[-1], params)
    elif "mock-performance-api" in url:
        return mock_performance_api(url.split("/")[-1], params)
    
    # Handle real HTTP APIs
    try:
        print(f"🌐 Making {method} request to: {url}")
        
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        if method.upper() == "GET":
            response = requests.get(url, params=params, headers=headers, timeout=30)
        elif method.upper() == "POST":
            response = requests.post(url, json=params, headers=headers, timeout=30)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        response.raise_for_status()
        
        try:
            return response.json()
        except:
            return {"text": response.text, "status": response.status_code}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API call failed: {e}")
        return {"error": str(e)}

def call_llm(prompt: str) -> str:
    """Call LLM API or return mock response for demo"""
    
    llm_url = os.getenv("LLM_API_URL")
    llm_key = os.getenv("LLM_API_KEY", "")
    
    if not llm_url:
        # Mock LLM responses for demo
        return mock_llm_response(prompt)
    
    try:
        headers = {"Content-Type": "application/json"}
        if llm_key:
            headers["Authorization"] = f"Bearer {llm_key}"
        
        payload = {
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.1
        }
        
        response = requests.post(llm_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return (
            result.get("response") or
            result.get("text") or
            result.get("choices", [{}])[0].get("text", "") or
            str(result)
        ).strip()
        
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        return mock_llm_response(prompt)

def mock_llm_response(prompt: str) -> str:
    """Generate mock LLM responses for demo purposes"""
    
    prompt_lower = prompt.lower()
    
    # Intent classification
    if "classify" in prompt_lower or "intent" in prompt_lower:
        if "account owner" in prompt_lower or "project owner" in prompt_lower:
            return "get_project_owner"
        elif "vacation" in prompt_lower or "leave" in prompt_lower:
            return "check_vacation_status"
        elif "employee" in prompt_lower and ("get" in prompt_lower or "show" in prompt_lower):
            return "get_employee"
        elif "meeting" in prompt_lower or "calendar" in prompt_lower:
            return "check_availability"
        elif "performance" in prompt_lower or "review" in prompt_lower:
            return "get_performance_review"
        elif "team" in prompt_lower:
            return "get_project_team"
        else:
            return "unknown"
    
    # Parameter extraction
    elif "extract" in prompt_lower or "parameter" in prompt_lower:
        params = {}
        
        # Extract employee info
        if "john doe" in prompt_lower or "john" in prompt_lower:
            params["employee_id"] = "E123"
            params["name"] = "John Doe"
        elif "sarah" in prompt_lower:
            params["employee_id"] = "E124"
            params["name"] = "Sarah Johnson"
        elif "jane" in prompt_lower:
            params["employee_id"] = "M456"
            params["name"] = "Jane Smith"
        
        # Extract project info
        if "111811" in prompt_lower:
            params["project_id"] = "111811"
        elif "222822" in prompt_lower:
            params["project_id"] = "222822"
        elif "alpha" in prompt_lower:
            params["project_id"] = "111811"
        elif "beta" in prompt_lower:
            params["project_id"] = "222822"
        
        # Extract specific employee IDs
        emp_match = re.search(r'[EM]\d+', prompt)
        if emp_match:
            params["employee_id"] = emp_match.group()
        
        return json.dumps(params)
    
    # Response formatting
    elif "format" in prompt_lower:
        if "john doe" in prompt_lower and "project alpha" in prompt_lower:
            return "John Doe is the Account Owner of Project Alpha (Seal ID: 111811). He is currently on vacation from December 20-30, 2024."
        elif "performance" in prompt_lower:
            return "John Doe's latest performance review (June 2024): Exceeds Expectations with a score of 4.2/5.0. He met 8 out of 10 goals and is scheduled for his next review on December 15, 2024."
        else:
            return "Based on the API data retrieved, here's the information you requested."
    
    return "Mock LLM response for demo purposes"

# =============================================================================
# MOCK API HANDLERS
# =============================================================================

def mock_employee_api(endpoint: str, params: Dict = None) -> Dict[str, Any]:
    """Mock employee API responses"""
    
    if endpoint == "get_employee" and params and "id" in params:
        emp_id = params["id"]
        return MOCK_EMPLOYEES.get(emp_id, {"error": f"Employee {emp_id} not found"})
    
    elif endpoint == "list_employees":
        return {"employees": list(MOCK_EMPLOYEES.values())}
    
    elif endpoint == "search_employees" and params:
        results = []
        search_term = params.get("name", "").lower()
        for emp in MOCK_EMPLOYEES.values():
            if search_term in emp["name"].lower():
                results.append(emp)
        return {"employees": results}
    
    return {"error": f"Unknown employee endpoint: {endpoint}"}

def mock_seal_api(endpoint: str, params: Dict = None) -> Dict[str, Any]:
    """Mock seal/project API responses"""
    
    if endpoint == "get_seal" and params and "id" in params:
        seal_id = params["id"]
        return MOCK_SEALS.get(seal_id, {"error": f"Seal {seal_id} not found"})
    
    elif endpoint == "get_account_owner" and params and "id" in params:
        seal_id = params["id"]
        if seal_id in MOCK_SEALS:
            seal = MOCK_SEALS[seal_id]
            ao_id = seal["account_owner_id"]
            if ao_id in MOCK_EMPLOYEES:
                return {
                    "project": seal["project_name"],
                    "account_owner": MOCK_EMPLOYEES[ao_id]
                }
    
    elif endpoint == "list_seals":
        return {"seals": list(MOCK_SEALS.values())}
    
    return {"error": f"Unknown seal endpoint: {endpoint}"}

def mock_leave_api(endpoint: str, params: Dict = None) -> Dict[str, Any]:
    """Mock employee leave API responses"""
    
    if endpoint == "check_vacation_status" and params and "id" in params:
        emp_id = params["id"]
        if emp_id in MOCK_LEAVES:
            leave_data = MOCK_LEAVES[emp_id]
            is_on_vacation = leave_data["current_leave"]["status"] == "on_leave"
            return {
                "employee_id": emp_id,
                "is_on_vacation": is_on_vacation,
                "current_leave": leave_data["current_leave"],
                "leave_balance": leave_data["leave_balance"]
            }
    
    elif endpoint == "get_current_leaves":
        on_leave = []
        for emp_id, leave_data in MOCK_LEAVES.items():
            if leave_data["current_leave"]["status"] == "on_leave":
                emp_info = MOCK_EMPLOYEES.get(emp_id, {})
                on_leave.append({
                    "employee": emp_info.get("name", emp_id),
                    "employee_id": emp_id,
                    "leave_type": leave_data["current_leave"]["type"],
                    "end_date": leave_data["current_leave"]["end_date"]
                })
        return {"employees_on_leave": on_leave}
    
    return {"error": f"Unknown leave endpoint: {endpoint}"}

def mock_calendar_api(endpoint: str, params: Dict = None) -> Dict[str, Any]:
    """Mock employee calendar API responses"""
    
    if endpoint == "check_availability" and params and "id" in params:
        emp_id = params["id"]
        if emp_id in MOCK_CALENDAR:
            meetings = MOCK_CALENDAR[emp_id]["today_meetings"]
            return {
                "employee_id": emp_id,
                "is_available": len(meetings) < 3,
                "today_meetings": meetings,
                "next_free_slot": "16:00-17:00"
            }
    
    return {"error": f"Unknown calendar endpoint: {endpoint}"}

def mock_performance_api(endpoint: str, params: Dict = None) -> Dict[str, Any]:
    """Mock performance review API responses"""
    
    if endpoint == "get_employee_review" and params and "id" in params:
        emp_id = params["id"]
        return MOCK_PERFORMANCE.get(emp_id, {"error": f"No performance data for {emp_id}"})
    
    return {"error": f"Unknown performance endpoint: {endpoint}"}

# =============================================================================
# API REGISTRY
# =============================================================================

REGISTERED_APIS = {}

def register_api(name: str, base_url: str, endpoints: Dict[str, Dict], auth_headers: Dict = None):
    """Register an API for use"""
    REGISTERED_APIS[name] = {
        "base_url": base_url.rstrip("/"),
        "endpoints": endpoints,
        "auth_headers": auth_headers or {}
    }
    print(f"✅ Registered API: {name}")

def setup_demo_apis():
    """Set up all demo APIs"""
    
    print("🔧 Setting up demo APIs...")
    
    # Employee API
    register_api(
        name="employee_api",
        base_url="https://mock-employee-api.com",
        endpoints={
            "get_employee": {"path": "/get_employee", "method": "GET"},
            "list_employees": {"path": "/list_employees", "method": "GET"},
            "search_employees": {"path": "/search_employees", "method": "GET"}
        }
    )
    
    # Seal/Project API
    register_api(
        name="seal_api",
        base_url="https://mock-seal-api.com",
        endpoints={
            "get_seal": {"path": "/get_seal", "method": "GET"},
            "list_seals": {"path": "/list_seals", "method": "GET"},
            "get_account_owner": {"path": "/get_account_owner", "method": "GET"}
        }
    )
    
    # Leave API
    register_api(
        name="leave_api",
        base_url="https://mock-leave-api.com",
        endpoints={
            "check_vacation_status": {"path": "/check_vacation_status", "method": "GET"},
            "get_current_leaves": {"path": "/get_current_leaves", "method": "GET"}
        }
    )
    
    # Calendar API
    register_api(
        name="calendar_api",
        base_url="https://mock-calendar-api.com",
        endpoints={
            "check_availability": {"path": "/check_availability", "method": "GET"}
        }
    )
    
    # Performance API (demonstrating easy addition)
    register_api(
        name="performance_api",
        base_url="https://mock-performance-api.com",
        endpoints={
            "get_employee_review": {"path": "/get_employee_review", "method": "GET"}
        }
    )

# =============================================================================
# LANGGRAPH WORKFLOW NODES
# =============================================================================

def classify_intent_node(state: RouterState) -> RouterState:
    """Classify the user's intent"""
    
    print(f"🎯 Classifying intent...")
    
    prompt = f"""
    Classify this question into one of these intents:
    - get_employee: Get employee information
    - get_project_owner: Find project account owner
    - check_vacation_status: Check vacation/leave status
    - check_availability: Check calendar availability
    - get_performance_review: Get performance review data
    - get_project_team: Get project team information
    - complex_query: Multiple API calls needed
    - unknown: Cannot determine
    
    Question: "{state['original_question']}"
    
    Respond with just the intent name:
    """
    
    intent = call_llm(prompt)
    state["intent"] = intent.strip()
    print(f"   ✅ Intent: {state['intent']}")
    
    return state

def extract_parameters_node(state: RouterState) -> RouterState:
    """Extract parameters from the question"""
    
    print(f"🔍 Extracting parameters...")
    
    prompt = f"""
    Extract parameters from this question:
    
    Question: "{state['original_question']}"
    Intent: {state['intent']}
    
    Look for employee IDs, names, project IDs, dates, etc.
    Return JSON only:
    """
    
    param_response = call_llm(prompt)
    
    try:
        # Try to parse JSON
        json_match = re.search(r'\{.*\}', param_response, re.DOTALL)
        if json_match:
            parameters = json.loads(json_match.group())
        else:
            parameters = {}
    except:
        parameters = {}
    
    state["parameters"] = parameters
    print(f"   ✅ Parameters: {parameters}")
    
    return state

def plan_api_calls_node(state: RouterState) -> RouterState:
    """Plan which APIs to call"""
    
    print(f"📋 Planning API calls...")
    
    intent = state["intent"]
    apis_to_call = []
    
    if intent == "get_employee":
        apis_to_call = ["employee_api"]
    elif intent == "get_project_owner":
        apis_to_call = ["seal_api", "employee_api"]
    elif intent == "check_vacation_status":
        apis_to_call = ["employee_api", "leave_api"]
    elif intent == "check_availability":
        apis_to_call = ["employee_api", "calendar_api"]
    elif intent == "get_performance_review":
        apis_to_call = ["employee_api", "performance_api"]
    elif intent == "get_project_team":
        apis_to_call = ["seal_api"]
    else:
        apis_to_call = ["employee_api"]  # Default
    
    state["apis_to_call"] = apis_to_call
    print(f"   ✅ Will call: {apis_to_call}")
    
    return state

def execute_api_calls_node(state: RouterState) -> RouterState:
    """Execute the planned API calls"""
    
    print(f"🚀 Executing API calls...")
    
    api_responses = {}
    parameters = state["parameters"]
    
    for api_name in state["apis_to_call"]:
        if api_name not in REGISTERED_APIS:
            continue
        
        # Choose endpoint based on API and parameters
        endpoint_name, endpoint_params = choose_endpoint(api_name, state["intent"], parameters)
        
        if endpoint_name:
            response = call_api_endpoint(api_name, endpoint_name, endpoint_params)
            api_responses[api_name] = response
            print(f"   ✅ Called {api_name}")
    
    state["api_responses"] = api_responses
    return state

def format_response_node(state: RouterState) -> RouterState:
    """Format the final response"""
    
    print(f"✨ Formatting response...")
    
    if not state["api_responses"]:
        state["final_answer"] = "Sorry, I couldn't retrieve any information."
        return state
    
    prompt = f"""
    Question: "{state['original_question']}"
    Intent: {state['intent']}
    
    API Data: {json.dumps(state['api_responses'], indent=2)}
    
    Format this into a clear, helpful answer:
    """
    
    answer = call_llm(prompt)
    state["final_answer"] = answer
    
    return state

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def choose_endpoint(api_name: str, intent: str, parameters: Dict[str, Any]) -> tuple:
    """Choose the appropriate endpoint for an API"""
    
    if api_name == "employee_api":
        if "employee_id" in parameters:
            return "get_employee", {"id": parameters["employee_id"]}
        elif "name" in parameters:
            return "search_employees", {"name": parameters["name"]}
        else:
            return "list_employees", {}
    
    elif api_name == "seal_api":
        if intent == "get_project_owner" and "project_id" in parameters:
            return "get_account_owner", {"id": parameters["project_id"]}
        elif "project_id" in parameters:
            return "get_seal", {"id": parameters["project_id"]}
        else:
            return "list_seals", {}
    
    elif api_name == "leave_api":
        if "employee_id" in parameters:
            return "check_vacation_status", {"id": parameters["employee_id"]}
        else:
            return "get_current_leaves", {}
    
    elif api_name == "calendar_api":
        if "employee_id" in parameters:
            return "check_availability", {"id": parameters["employee_id"]}
    
    elif api_name == "performance_api":
        if "employee_id" in parameters:
            return "get_employee_review", {"id": parameters["employee_id"]}
    
    return None, {}

def call_api_endpoint(api_name: str, endpoint_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Call a specific API endpoint"""
    
    api_info = REGISTERED_APIS[api_name]
    endpoint_info = api_info["endpoints"][endpoint_name]
    
    base_url = api_info["base_url"]
    path = endpoint_info["path"]
    method = endpoint_info["method"]
    
    # Replace path parameters
    for key, value in params.items():
        path = path.replace(f"{{{key}}}", str(value))
        path = path.replace("{id}", str(value))
    
    full_url = f"{base_url}{path}"
    headers = api_info["auth_headers"]
    
    return make_api_call(full_url, method, params, headers)

# =============================================================================
# LANGGRAPH WORKFLOW OR SIMPLE WORKFLOW
# =============================================================================

if LANGGRAPH_AVAILABLE:
    # Use LangGraph workflow
    def create_workflow():
        """Create LangGraph workflow"""
        workflow = StateGraph(RouterState)
        
        workflow.add_node("classify_intent", classify_intent_node)
        workflow.add_node("extract_parameters", extract_parameters_node)
        workflow.add_node("plan_api_calls", plan_api_calls_node)
        workflow.add_node("execute_api_calls", execute_api_calls_node)
        workflow.add_node("format_response", format_response_node)
        
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "extract_parameters")
        workflow.add_edge("extract_parameters", "plan_api_calls")
        workflow.add_edge("plan_api_calls", "execute_api_calls")
        workflow.add_edge("execute_api_calls", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    class APIRouter:
        def __init__(self):
            self.workflow = create_workflow()
            print("✅ LangGraph workflow initialized")
        
        def ask_question(self, question: str) -> str:
            initial_state = RouterState(
                original_question=question,
                intent="",
                parameters={},
                apis_to_call=[],
                api_responses={},
                final_answer="",
                error=None
            )
            
            try:
                final_state = self.workflow.invoke(initial_state)
                return final_state["final_answer"]
            except Exception as e:
                return f"Error: {str(e)}"

else:
    # Fallback simple workflow without LangGraph
    class APIRouter:
        def __init__(self):
            print("✅ Simple workflow initialized (LangGraph not available)")
        
        def ask_question(self, question: str) -> str:
            try:
                # Simple sequential execution
                state = {
                    "original_question": question,
                    "intent": "",
                    "parameters": {},
                    "apis_to_call": [],
                    "api_responses": {},
                    "final_answer": ""
                }
                
                # Execute workflow steps
                state = classify_intent_node(state)
                state = extract_parameters_node(state)
                state = plan_api_calls_node(state)
                state = execute_api_calls_node(state)
                state = format_response_node(state)
                
                return state["final_answer"]
                
            except Exception as e:
                return f"Error: {str(e)}"

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("=" * 70)
    print("🚀 Complete API Router - Ready to Run!")
    print("=" * 70)
    
    # Setup
    setup_demo_apis()
    router = APIRouter()
    
    print(f"\n📊 Available APIs: {list(REGISTERED_APIS.keys())}")
    print(f"📊 Mock Data: {len(MOCK_EMPLOYEES)} employees, {len(MOCK_SEALS)} projects")
    
    # Demo questions
    demo_questions = [
        "Get employee E123",
        "Who is the account owner of project 111811?",
        "Is John Doe on vacation?",
        "Show me John's performance review",
        "Is Sarah available for a meeting?",
        