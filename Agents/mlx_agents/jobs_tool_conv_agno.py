import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools import tool
from agno.tools.user_control_flow import UserControlFlowTools
from agno.workflow import Workflow
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from bs4 import BeautifulSoup
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL = "mlx-community/Qwen3-4B-Instruct-2507-bf16"
MAX_PAGES = 15
DEFAULT_REQUEST = "Find jobs on dev.bg."


def parse_bg_date(date_text: str) -> datetime:
    """Parse Bulgarian date format like '18 <bg-month>' to datetime object."""
    bg_months = {
        "\u044f\u043d.": "01",
        "\u0444\u0435\u0432\u0440\u0443\u0430\u0440\u0438": "02",
        "\u043c\u0430\u0440\u0442": "03",
        "\u0430\u043f\u0440\u0438\u043b": "04",
        "\u043c\u0430\u0439": "05",
        "\u044e\u043d\u0438": "06",
        "\u044e\u043b\u0438": "07",
        "\u0430\u0432\u0433\u0443\u0441\u0442": "08",
        "\u0441\u0435\u043f\u0442\u0435\u043c\u0432\u0440\u0438": "09",
        "\u043e\u043a\u0442\u043e\u043c\u0432\u0440\u0438": "10",
        "\u043d\u043e\u0435\u043c\u0432\u0440\u0438": "11",
        "\u0434\u0435\u043a.": "12",
    }

    date_parts = date_text.strip().split()
    if len(date_parts) != 2:
        raise ValueError(f"Invalid date format: {date_text}")

    day, month = date_parts
    current_year = datetime.now().year

    month_num = bg_months.get(month.lower())
    if not month_num:
        raise ValueError(f"Invalid month: {month}")

    date_str = f"{current_year}-{month_num}-{day.zfill(2)}"
    return datetime.strptime(date_str, "%Y-%m-%d")


def parse_date(date_str: str) -> datetime:
    """Parse date string and return datetime object."""
    if date_str.lower() == "today":
        return datetime.now()
    if date_str.lower() == "yesterday":
        return datetime.now() - timedelta(days=1)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.now()


def get_category_mapping() -> Dict[str, str]:
    """Map common category names to dev.bg category parameters."""
    return {
        "data science": "data-science",
        "machine learning": "data-science",
        "data": "data-science",
        "backend development": "back-end-development",
        "python development": "python",
    }


def scrape_dev_bg_jobs(category: str, target_date: datetime):
    """Scrape jobs from dev.bg for a specific category and date."""
    try:
        category_mapping = get_category_mapping()
        category_param = category_mapping.get(
            category.lower(), category.lower().replace(" ", "-")
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        job_listings = []
        for page in range(1, MAX_PAGES):
            base_url = f"https://dev.bg/company/jobs/{category_param}?_paged={page}"
            response = requests.get(base_url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            job_containers = soup.find_all(
                "div", class_=lambda x: x and x.startswith("job-list-item")
            )

            target_date_str = target_date.strftime("%Y-%m-%d")

            for job in job_containers:
                try:
                    title_elem = job.find(
                        "h6",
                        class_=lambda x: x and "job-title" in x,
                    )
                    title = (
                        title_elem.get_text(strip=True)
                        if title_elem
                        else "Title not found"
                    )

                    company_elem = job.find(
                        ["span", "div", "p"],
                        class_=re.compile(r"company|employer", re.I),
                    )
                    company = (
                        company_elem.get_text(strip=True)
                        if company_elem
                        else "Company not specified"
                    )

                    date_elem = job.find("span", class_="date")
                    date_posted = (
                        date_elem.get_text(strip=True)
                        if date_elem
                        else "Date not found"
                    )
                    parsed_date = parse_bg_date(date_posted)
                    formatted_date = parsed_date.strftime("%Y-%m-%d")

                    link_elem = job.find("a")
                    link = link_elem.get("href") if link_elem else ""
                    if link and not link.startswith("http"):
                        link = f"https://dev.bg{link}"

                    job_info = {
                        "title": title,
                        "company": company,
                        "date_posted": date_posted,
                        "link": link,
                        "category": category,
                    }
                    if target_date_str == formatted_date:
                        job_listings.append(job_info)

                except Exception as e:
                    log.error(f"Error parsing job: {e}")
                    continue

        return job_listings

    except requests.RequestException as e:
        return f"Error fetching jobs from dev.bg: {str(e)}"
    except Exception as e:
        return f"Error processing jobs data: {str(e)}"


def get_todays_jobs_data(category: str, date: str = "today") -> str:
    """Get jobs posted today or on a given date in a specific category from dev.bg."""
    try:
        target_date = parse_date(date)
        jobs = scrape_dev_bg_jobs(category, target_date)

        if isinstance(jobs, str):
            return jobs

        if not jobs:
            return (
                f"No jobs found for category '{category}' on "
                f"{target_date.strftime('%Y-%m-%d')} on dev.bg"
            )

        result = (
            f"Found {len(jobs)} jobs in '{category}' category for "
            f"{target_date.strftime('%Y-%m-%d')}:\n\n"
        )

        for i, job in enumerate(jobs, 1):
            result += f"{i}. {job['title']}\n"
            result += f"   Company: {job['company']}\n"
            result += f"   Posted: {job['date_posted']}\n"
            if job["link"]:
                result += f"   Link: {job['link']}\n"
            result += "\n"

        return result

    except Exception as e:
        return f"Error getting jobs: {str(e)}"


def create_panel(content, title, border_style="blue"):
    from rich.box import HEAVY
    from rich.panel import Panel

    return Panel(
        content,
        title=title,
        title_align="left",
        border_style=border_style,
        box=HEAVY,
        expand=True,
        padding=(1, 1),
    )


def format_step_content_for_display(step_output: StepOutput) -> str:
    actual_content = step_output.content
    if not actual_content:
        return ""
    if isinstance(actual_content, str):
        return actual_content
    if isinstance(actual_content, dict):
        return f"**Structured Output:**\n\n```json\n{json.dumps(actual_content, indent=2, default=str)}\n```"
    return str(actual_content)


def print_step_output_recursive(
    step_output: StepOutput,
    step_number: int,
    markdown: bool,
    console: Console,
    depth: int = 0,
) -> None:
    if step_output.content:
        formatted_content = format_step_content_for_display(step_output)
        if depth == 0:
            title = f"Step {step_number}: {step_output.step_name} (Completed)"
        else:
            title = f"{'  ' * depth}└─ {step_output.step_name} (Completed)"
        step_panel = create_panel(
            content=Markdown(formatted_content) if markdown else formatted_content,
            title=title,
            border_style="orange3",
        )
        console.print(step_panel)

    if step_output.steps:
        for j, nested_step in enumerate(step_output.steps):
            print_step_output_recursive(
                nested_step, j + 1, markdown, console, depth + 1
            )


def render_workflow_output(
    workflow: Workflow,
    workflow_response,
    input_payload: Dict[str, Any],
    markdown: bool = True,
) -> None:
    console = Console()
    workflow_info = f"**Workflow:** {workflow.name}"
    if workflow.description:
        workflow_info += f"\n\n**Description:** {workflow.description}"
    workflow_info += f"\n\n**Steps:** {workflow._get_step_count()} steps"
    if input_payload:
        data_display = json.dumps(input_payload, indent=2, default=str)
        workflow_info += f"\n\n**Structured Input:**\n```json\n{data_display}\n```"

    workflow_panel = create_panel(
        content=Markdown(workflow_info) if markdown else workflow_info,
        title="Workflow Information",
        border_style="cyan",
    )
    console.print(workflow_panel)

    step_results = getattr(workflow_response, "step_results", None) or []
    if step_results:
        for i, step_output in enumerate(step_results):
            print_step_output_recursive(step_output, i + 1, markdown, console)
    elif workflow_response.content:
        content = workflow_response.content
        rendered = Markdown(str(content)) if markdown else str(content)
        console.print(
            create_panel(
                content=rendered,
                title="Workflow Result",
                border_style="orange3",
            )
        )


@tool(stop_after_tool_call=False)
def run_job_search_workflow(category: str, date: str = "today") -> str:
    """Run the job search workflow and return the formatted results."""
    input_payload = normalize_search_request({"category": category, "date": date})
    if not input_payload["category"].strip():
        return "Missing job category. Please provide a category."

    workflow = build_workflow()
    workflow_response = workflow.run(input=input_payload)
    render_workflow_output(workflow, workflow_response, input_payload)
    if workflow_response.content is None:
        return "Workflow completed with no results."
    return str(workflow_response.content)


def build_session_agent() -> Agent:
    return Agent(
        instructions=[
            "You are an interactive job search assistant for dev.bg.",
            "When you need category or date, call get_user_input for only the missing fields.",
            "Once you have category and date, call run_job_search_workflow.",
            "Only ask for another search if the user requests more or you believe a follow-up is useful.",
            "If you ask for another search, first ask via get_user_input, then request new criteria if needed.",
            "Include the tool results in your response and keep commentary brief.",
        ],
        tools=[UserControlFlowTools(), run_job_search_workflow],
        model=OpenAILike(
            id=MODEL,
            api_key="not-needed",
            base_url="http://localhost:8080/v1",
            temperature=0.2,
            max_tokens=2000,
            max_completion_tokens=2000,
        ),
    )


def extract_search_request_from_text(text: str) -> Dict[str, str]:
    if not text:
        return {}

    text_lower = text.lower()
    category = None
    date = None

    match = re.search(
        r"\bin\s+([A-Za-z][A-Za-z0-9&/\- ]+?)\s+category\b",
        text,
        re.IGNORECASE,
    )
    if match:
        category = match.group(1).strip()
    else:
        category_keywords = {
            "data science": "Data Science",
            "machine learning": "Machine Learning",
            "backend development": "Backend Development",
            "back-end development": "Backend Development",
            "backend": "Backend Development",
            "python development": "Python Development",
            "python": "Python Development",
        }
        for key, label in category_keywords.items():
            if key in text_lower:
                category = label
                break

    if "today" in text_lower:
        date = "today"
    elif "yesterday" in text_lower:
        date = "yesterday"
    else:
        date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", text)
        if date_match:
            date = date_match.group(0)

    extracted: Dict[str, str] = {}
    if category:
        extracted["category"] = category
    if date:
        extracted["date"] = date
    return extracted


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1))
        except json.JSONDecodeError:
            return None

    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            return None

    return None


def normalize_search_request(data: Dict[str, Any]) -> Dict[str, str]:
    category = str(data.get("category") or "").strip()
    date = str(data.get("date") or "").strip() or "today"
    return {"category": category, "date": date}


def iter_user_input_requirements(run_response) -> List[Any]:
    active = getattr(run_response, "active_requirements", None)
    if active is None:
        active = getattr(run_response, "requirements", None) or []
    return [req for req in active if getattr(req, "needs_user_input", False)]


def apply_prefill(run_response, prefill: Dict[str, str]) -> None:
    for requirement in iter_user_input_requirements(run_response):
        for field in requirement.user_input_schema or []:
            if field.value is None and field.name in prefill:
                field.value = prefill[field.name]


def render_user_input_request(run_response) -> None:
    console = Console()
    header = getattr(run_response, "content", None) or "Run paused. User input is required."
    lines = [header, "", "Required fields:"]

    for requirement in iter_user_input_requirements(run_response):
        for field in requirement.user_input_schema or []:
            line = f"- {field.name}"
            if field.description:
                line += f": {field.description}"
            if field.value is not None:
                line += f" (provided: {field.value})"
            lines.append(line)

    panel = create_panel(
        content=Markdown("\n".join(lines)),
        title="Run Paused",
        border_style="blue",
    )
    console.print(panel)


def run_agent_with_user_input(agent: Agent, prompt: str) -> Any:
    prefill = extract_search_request_from_text(prompt)
    run_response = agent.run(prompt)

    if run_response.is_paused and prefill:
        apply_prefill(run_response, prefill)
        if not iter_user_input_requirements(run_response):
            run_response = agent.continue_run(
                run_response=run_response, requirements=run_response.requirements
            )

    while run_response.is_paused:
        render_user_input_request(run_response)
        for requirement in iter_user_input_requirements(run_response):
            for field in requirement.user_input_schema or []:
                if field.value is not None:
                    continue
                prompt_text = field.name
                if field.description:
                    prompt_text += f" ({field.description})"
                field.value = Prompt.ask(prompt_text)

        run_response = agent.continue_run(
            run_response=run_response, requirements=run_response.requirements
        )

    return run_response


def get_response_text(run_response) -> str:
    content = getattr(run_response, "content", None)
    if content:
        return str(content)
    messages = getattr(run_response, "messages", None) or []
    for message in reversed(messages):
        if getattr(message, "role", None) == "assistant" and message.content:
            return str(message.content)
    return ""


def format_search_request(step_input: StepInput) -> StepOutput:
    input_data: Dict[str, Any] = {}
    if isinstance(step_input.input, dict):
        input_data = step_input.input
    elif isinstance(step_input.input, str):
        input_data = parse_json_from_text(step_input.input) or {}

    normalized = normalize_search_request(input_data)
    return StepOutput(content=json.dumps(normalized, indent=2))


def fetch_jobs(step_input: StepInput) -> StepOutput:
    formatted = step_input.get_step_content("Format Search Request")
    normalized: Dict[str, Any] = {}

    if isinstance(formatted, dict):
        normalized = formatted
    elif isinstance(formatted, str):
        normalized = parse_json_from_text(formatted) or {}

    normalized = normalize_search_request(normalized)
    category = normalized.get("category", "").strip()
    date = normalized.get("date", "today")
    if not category:
        return StepOutput(content="Missing job category. Please try again.")

    return StepOutput(content=get_todays_jobs_data(category=category, date=date))


def build_workflow() -> Workflow:
    return Workflow(
        name="Job Search Workflow",
        description="Normalize job search criteria and fetch dev.bg results.",
        steps=[
            Step(name="Format Search Request", executor=format_search_request),
            Step(name="Fetch Jobs", executor=fetch_jobs),
        ],
    )


if __name__ == "__main__":
    user_input = Prompt.ask("Job search request", default=DEFAULT_REQUEST)
    agent = build_session_agent()
    run_response = run_agent_with_user_input(agent, user_input)
    response_text = get_response_text(run_response)
    if response_text:
        console = Console()
        console.print(
            create_panel(
                content=Markdown(response_text),
                title="Assistant Response",
                border_style="green",
            )
        )
