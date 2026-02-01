import os
import sys
from datetime import datetime, timedelta
from typing import List, TypedDict

import agentlightning as agl
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.tools.user_control_flow import UserControlFlowTools
from jobs_tool_conv_agno import (
    DEFAULT_REQUEST,
    MODEL,
    get_response_text,
    get_todays_jobs_data,
    normalize_search_request,
    run_agent_with_user_input,
)
from openai import AsyncOpenAI
from opentelemetry import trace

DEFAULT_SYSTEM_PROMPT = "\n".join(
    [
        "You are an interactive job search assistant for dev.bg.",
        "Never ask for missing details directly; always use get_user_input.",
        "When you need category or date, call get_user_input for only the missing fields.",
        "Once you have category and date, call search_devbg_jobs.",
        "Only ask for another search if the user requests more or you believe a follow-up is useful.",
        "If you ask for another search, first ask via get_user_input, then request new criteria if needed.",
        "Include the tool results in your response and keep commentary brief.",
    ]
)
DEFAULT_OPENAI_BASE_URL = "http://localhost:8080/v1"
APO_DEFAULT_BEAM_ROUNDS = 3
APO_DEFAULT_BEAM_WIDTH = 4
APO_DEFAULT_BRANCH_FACTOR = 4
APO_DEFAULT_GRADIENT_BATCH_SIZE = 4
APO_DEFAULT_VAL_BATCH_SIZE = 16
APO_DEFAULT_DIVERSITY_TEMPERATURE = 1.0


class JobSearchTask(TypedDict):
    request: str
    category: str
    date: str


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _select_execution(n_runners: int) -> tuple[int, str | None]:
    """Pick an execution strategy that avoids multiprocessing issues on macOS."""
    if sys.platform == "darwin":
        return 1, "shm"
    return n_runners, None


def _normalize_date(date_str: str) -> str:
    if date_str.lower() == "today":
        return datetime.now().strftime("%Y-%m-%d")
    if date_str.lower() == "yesterday":
        return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return date_str


def _format_prompt(prompt_template: agl.PromptTemplate, task: JobSearchTask) -> str:
    try:
        return prompt_template.format(**task)
    except (KeyError, ValueError):
        try:
            return prompt_template.template.format_map(_SafeFormatDict(task))
        except (KeyError, ValueError):
            return prompt_template.template


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number") from exc


@tool(stop_after_tool_call=False)
def search_devbg_jobs(category: str, date: str = "today") -> str:
    """Fetch dev.bg jobs for a category and date."""
    payload = normalize_search_request({"category": category, "date": date})
    if not payload["category"].strip():
        return "Missing job category. Please provide a category."
    return get_todays_jobs_data(category=payload["category"], date=payload["date"])


def build_session_agent(system_prompt: str) -> Agent:
    return Agent(
        instructions=[system_prompt],
        tools=[UserControlFlowTools(), search_devbg_jobs],
        model=OpenAILike(
            id=MODEL,
            api_key="not-needed",
            base_url="http://localhost:8080/v1",
            temperature=0.2,
            max_tokens=2000,
            max_completion_tokens=2000,
        ),
    )


def _emit_gen_ai_span(prompt: str, response: str) -> None:
    # Emit a minimal gen_ai span so APO's TraceToMessages can reconstruct messages.
    tracer = trace.get_tracer("jobs_tool_conv_agno_al")
    with tracer.start_as_current_span("openai.chat.completion") as span:
        span.set_attribute("gen_ai.prompt.0.role", "user")
        span.set_attribute("gen_ai.prompt.0.content", prompt)
        span.set_attribute("gen_ai.completion.0.role", "assistant")
        span.set_attribute("gen_ai.completion.0.content", response)


def _score_response(response_text: str, task: JobSearchTask) -> float:
    if not response_text:
        return 0.0

    lowered = response_text.lower()
    if "missing job category" in lowered or "error" in lowered:
        return 0.0

    reward = 0.0
    if task["category"].lower() in lowered:
        reward += 0.3

    expected_date = _normalize_date(task["date"])
    if expected_date in response_text:
        reward += 0.3

    if "found" in lowered or "no jobs found" in lowered:
        reward += 0.2

    if "company:" in lowered:
        reward += 0.2

    return min(reward, 1.0)


@agl.rollout
def job_search_rollout(
    task: JobSearchTask, prompt_template: agl.PromptTemplate
) -> float:
    system_prompt = _format_prompt(prompt_template, task)
    agent = build_session_agent(system_prompt)

    # Ensure request text contains category/date so the agent does not pause.
    request = task.get("request") or ""
    if not request.strip():
        request = f"Find jobs in {task['category']} category for {task['date']}."

    # Use the existing helper so any get_user_input requirement is auto-filled.
    run_response = run_agent_with_user_input(agent, request)
    response_text = get_response_text(run_response)

    _emit_gen_ai_span(request, response_text)
    return _score_response(response_text, task)


TRAIN_TASKS: List[JobSearchTask] = [
    {
        "request": "Find jobs in Data Science category today.",
        "category": "Data Science",
        "date": "today",
    },
    {
        "request": "Find jobs in Python Development category yesterday.",
        "category": "Python Development",
        "date": "yesterday",
    },
    {
        "request": "Find jobs in Backend Development category today.",
        "category": "Backend Development",
        "date": "today",
    },
]

VAL_TASKS: List[JobSearchTask] = [
    {
        "request": "Find jobs in Machine Learning category today.",
        "category": "Machine Learning",
        "date": "today",
    },
    {
        "request": "Find jobs in Data Science category yesterday.",
        "category": "Data Science",
        "date": "yesterday",
    },
]


def run_dev() -> None:
    n_runners, strategy = _select_execution(1)
    resource = agl.PromptTemplate(template=DEFAULT_SYSTEM_PROMPT, engine="f-string")
    trainer = agl.Trainer(
        n_runners=n_runners,
        strategy=strategy,
        tracer=agl.OtelTracer(),
        initial_resources={"main_prompt": resource},
    )
    trainer.dev(job_search_rollout, TRAIN_TASKS)


def run_apo() -> None:
    requested_runners = int(os.environ.get("AGL_N_RUNNERS", "1"))
    n_runners, strategy = _select_execution(requested_runners)
    base_url = (
        os.environ.get("APO_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or DEFAULT_OPENAI_BASE_URL
    )
    api_key = os.environ.get("APO_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key and base_url.startswith(("http://localhost", "http://127.0.0.1")):
        api_key = "not-needed"
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY (or APO_API_KEY) for APO.")

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    gradient_model = os.environ.get("APO_GRADIENT_MODEL") or MODEL
    apply_edit_model = os.environ.get("APO_EDIT_MODEL") or MODEL
    beam_rounds = _env_int("APO_BEAM_ROUNDS")
    beam_width = _env_int("APO_BEAM_WIDTH")
    branch_factor = _env_int("APO_BRANCH_FACTOR")
    gradient_batch_size = _env_int("APO_GRADIENT_BATCH_SIZE")
    val_batch_size = _env_int("APO_VAL_BATCH_SIZE")
    diversity_temperature = _env_float("APO_DIVERSITY_TEMPERATURE")

    algo_kwargs = {
        "gradient_model": gradient_model,
        "apply_edit_model": apply_edit_model,
        "beam_rounds": (
            beam_rounds if beam_rounds is not None else APO_DEFAULT_BEAM_ROUNDS
        ),
        "beam_width": beam_width if beam_width is not None else APO_DEFAULT_BEAM_WIDTH,
        "branch_factor": (
            branch_factor if branch_factor is not None else APO_DEFAULT_BRANCH_FACTOR
        ),
        "gradient_batch_size": (
            gradient_batch_size
            if gradient_batch_size is not None
            else APO_DEFAULT_GRADIENT_BATCH_SIZE
        ),
        "val_batch_size": (
            val_batch_size if val_batch_size is not None else APO_DEFAULT_VAL_BATCH_SIZE
        ),
        "diversity_temperature": (
            diversity_temperature
            if diversity_temperature is not None
            else APO_DEFAULT_DIVERSITY_TEMPERATURE
        ),
    }
    algo = agl.APO(openai_client, **algo_kwargs)
    resource = agl.PromptTemplate(template=DEFAULT_SYSTEM_PROMPT, engine="f-string")

    trainer = agl.Trainer(
        algorithm=algo,
        n_runners=n_runners,
        strategy=strategy,
        tracer=agl.OtelTracer(),
        adapter=agl.TraceToMessages(),
        initial_resources={"main_prompt": resource},
    )
    trainer.fit(job_search_rollout, train_dataset=TRAIN_TASKS, val_dataset=VAL_TASKS)

    best = algo.get_best_prompt()
    print("Best prompt template:")
    print(best.template)


def run_single(prompt: str) -> None:
    agent = build_session_agent(DEFAULT_SYSTEM_PROMPT)
    run_response = run_agent_with_user_input(agent, prompt)
    print(get_response_text(run_response))


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if mode == "dev":
        run_dev()
        return
    if mode == "apo":
        run_apo()
        return
    if mode == "single":
        prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_REQUEST
        run_single(prompt)
        return

    raise SystemExit("Usage: python jobs_tool_conv_agno_al.py [dev|apo|single]")


if __name__ == "__main__":
    main()
