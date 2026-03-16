import argparse
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai.like import OpenAILike
from agno.os import AgentOS
from agno.tools import tool
from agno.tools.user_control_flow import UserControlFlowTools
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
AGENT_ID = "dev-bg-job-search-agent"
AGENTOS_ID = "mlx-job-search-agentos"
AGENTOS_HOST = "127.0.0.1"
AGENTOS_PORT = 7777
AGNO_RUNTIME_DIR = Path(__file__).resolve().parent / ".agno"
AGNO_DB_FILE = AGNO_RUNTIME_DIR / "jobs_tool_conv_agno.db"


def build_shared_db() -> SqliteDb:
    AGNO_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return SqliteDb(db_file=str(AGNO_DB_FILE))


AGNO_DB = build_shared_db()


def parse_bg_date(date_text: str) -> datetime:
    """Parse Bulgarian date format like '18 <bg-month>' to datetime object."""
    bg_months = {
        "ян.": "01",
        "фев.": "02",
        "мар.": "03",
        "април": "04",
        "май": "05",
        "юни": "06",
        "юли": "07",
        "август": "08",
        "септември": "09",
        "октомври": "10",
        "ноември": "11",
        "дек.": "12",
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


def run_job_search(category: str, date: str = "today") -> str:
    input_payload = normalize_search_request({"category": category, "date": date})
    if not input_payload["category"].strip():
        return "Missing job category. Please provide a category."
    return get_todays_jobs_data(
        category=input_payload["category"], date=input_payload["date"]
    )


def resolve_search_request(
    request: str = "", category: str = "", date: str = ""
) -> Dict[str, str]:
    extracted = extract_search_request_from_text(request)
    payload = {
        "category": category.strip() or extracted.get("category", ""),
        "date": date.strip() or extracted.get("date", "today"),
    }
    return normalize_search_request(payload)


@tool(stop_after_tool_call=False)
def search_devbg_jobs(category: str, date: str = "today") -> str:
    """Search dev.bg jobs for a category and date."""
    return run_job_search(category=category, date=date)


@tool(show_result=True, stop_after_tool_call=True)
def search_jobs_from_request(request: str, category: str = "", date: str = "") -> str:
    """Parse a free-form dev.bg job search request and return the matching jobs."""
    payload = resolve_search_request(request=request, category=category, date=date)
    if not payload["category"].strip():
        return (
            "I could not determine the job category from your request. "
            "Please include a category such as Data Science, Machine Learning, "
            "Backend Development, or Python Development."
        )
    return run_job_search(category=payload["category"], date=payload["date"])


def build_session_agent(db: Optional[SqliteDb] = None) -> Agent:
    return Agent(
        id=AGENT_ID,
        name="dev.bg Job Search Assistant",
        description="Search dev.bg jobs by category and date using the local MLX model server.",
        db=db or AGNO_DB,
        markdown=True,
        add_history_to_context=True,
        num_history_runs=3,
        instructions=[
            "You are an interactive job search assistant for dev.bg.",
            "Never ask for missing details directly; always use get_user_input.",
            "If the user's message already contains the category or date, extract those values yourself.",
            "When you need category or date, call get_user_input for only the missing fields.",
            "Once you have category and date, call search_devbg_jobs.",
            "Only ask for another search if the user requests more or you believe a follow-up is useful.",
            "If you ask for another search, first ask via get_user_input, then request new criteria if needed.",
            "Include the tool results in your response and keep commentary brief.",
        ],
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


def build_agent_os_agent(db: Optional[SqliteDb] = None) -> Agent:
    return Agent(
        id=AGENT_ID,
        name="dev.bg Job Search Assistant",
        description="Search dev.bg jobs by category and date using the local MLX model server.",
        db=db or AGNO_DB,
        markdown=True,
        add_history_to_context=True,
        num_history_runs=3,
        tool_call_limit=1,
        instructions=[
            "You are a dev.bg job search assistant.",
            "For job-search requests, call search_jobs_from_request using the user's full message as the request argument.",
            "Do not ask follow-up questions before calling search_jobs_from_request.",
            "Only ask a follow-up question if the tool result says it could not determine the category.",
            "Keep any extra commentary brief.",
        ],
        tools=[search_jobs_from_request],
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


def get_missing_user_inputs(run_response) -> List[Any]:
    missing_fields: List[Any] = []
    for requirement in iter_user_input_requirements(run_response):
        for field in requirement.user_input_schema or []:
            if field.value is None:
                missing_fields.append(field)
    return missing_fields


def render_user_input_request(run_response) -> None:
    missing_fields = get_missing_user_inputs(run_response)
    if not missing_fields:
        return

    console = Console()
    header = (
        getattr(run_response, "content", None) or "Run paused. User input is required."
    )
    lines = [header, "", "Required fields:"]

    for field in missing_fields:
        line = f"- {field.name}"
        if field.description:
            line += f": {field.description}"
        lines.append(line)

    panel = create_panel(
        content=Markdown("\n".join(lines)),
        title="Run Paused",
        border_style="blue",
    )
    console.print(panel)


def run_agent_with_user_input(agent: Agent, prompt: str) -> Any:
    prefill = extract_search_request_from_text(prompt)
    required_fields = {"category", "date"}
    missing_fields = required_fields - set(prefill.keys())

    previous_tool_choice = agent.tool_choice
    if missing_fields:
        agent.tool_choice = {"type": "function", "function": {"name": "get_user_input"}}

    try:
        run_response = agent.run(prompt)
    finally:
        agent.tool_choice = previous_tool_choice

    if missing_fields and not run_response.is_paused:
        nudge = (
            "Missing required fields: "
            + ", ".join(sorted(missing_fields))
            + ". Use get_user_input to request ONLY those fields. "
            + "Do not ask in plain text."
        )
        previous_tool_choice = agent.tool_choice
        agent.tool_choice = {"type": "function", "function": {"name": "get_user_input"}}
        try:
            run_response = agent.run(f"{prompt}\n\n{nudge}")
        finally:
            agent.tool_choice = previous_tool_choice

    if run_response.is_paused and prefill:
        apply_prefill(run_response, prefill)
        if not get_missing_user_inputs(run_response):
            run_response = agent.continue_run(
                run_response=run_response, requirements=run_response.requirements
            )

    while run_response.is_paused:
        if not get_missing_user_inputs(run_response):
            run_response = agent.continue_run(
                run_response=run_response, requirements=run_response.requirements
            )
            continue

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


def build_agent_os() -> AgentOS:
    return AgentOS(
        id=AGENTOS_ID,
        name="MLX Job Search AgentOS",
        description="AgentOS app for the dev.bg job search assistant.",
        db=AGNO_DB,
        agents=[build_agent_os_agent(db=AGNO_DB)],
    )


agent_os = build_agent_os()
# Expose a FastAPI app so AgentOS can be served with `fastapi dev`.
app = agent_os.get_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dev.bg Agno agent in the terminal or serve it via AgentOS."
    )
    parser.add_argument(
        "--agentos",
        action="store_true",
        help="Serve the AgentOS UI/API instead of running the terminal prompt.",
    )
    parser.add_argument(
        "--host",
        default=AGENTOS_HOST,
        help=f"Host to bind the AgentOS server to. Defaults to {AGENTOS_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=AGENTOS_PORT,
        help=f"Port to bind the AgentOS server to. Defaults to {AGENTOS_PORT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.agentos:
        agent_os.serve(app=app, host=args.host, port=args.port)
        return

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


if __name__ == "__main__":
    main()
