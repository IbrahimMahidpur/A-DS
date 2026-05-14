import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    _files_per_step: List[List[str]]  # Tracks files created per step
    user_query: str
    uploaded_files: List[str]
    _routing_flags: Dict[str, bool]
    parsed_documents: List[Dict]
    image_embeddings: Annotated[List[Any], operator.add]
    audio_transcripts: Annotated[List[str], operator.add]
    tabular_summaries: Annotated[List[Dict], operator.add]
    statistical_report: Dict[str, Any]
    analysis_plan: str
    analysis_tasks: List[Dict]
    hypotheses: List[str]
    current_step: int
    steps_total: int
    code_outputs: Annotated[List[str], operator.add]
    full_code_outputs: Annotated[List[str], operator.add]
    visualizations: Annotated[List[str], operator.add]
    saved_artifacts: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    reflections: Annotated[List[Dict], operator.add]
    retry_count: int
    vector_store_id: str
    retrieved_context: str
    eval_report: Dict[str, Any]
    final_report: str
    problem_spec: Dict[str, Any]
    model_selection: Dict[str, Any]
    ensemble_code_template: str
    session_id: str
    _last_task_name: str
    _last_files_created: List[str]
    _last_success: bool
    tuning_results: Dict[str, Any]
    gate_passed: bool
    messages: Annotated[list, add_messages]
    gate_reasons: List[str]
    reflection_report: Dict[str, Any]
    executive_summary: str
    business_recommendations: List[str]
