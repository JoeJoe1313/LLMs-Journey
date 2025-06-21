import json
import logging
import re
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

MODEL = "mlx-community/Qwen3-8B-8bit"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_todays_jobs",
            "description": "Get the jobs posted today or on a given date in a specific job category from dev.bg",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The job category (e.g., 'Data Science', 'Software Development', 'DevOps', etc.)",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date to search for jobs (e.g., 'today', '2024-06-15', 'yesterday'). Defaults to 'today'",
                        "default": "today",
                    },
                },
                "required": ["category"],
            },
        },
    }
]


def parse_bg_date(date_text: str) -> datetime:
    """Parse Bulgarian date format like '18 юни' to datetime object"""
    bg_months = {
        # "януари": "01",
        # "февруари": "02",
        # "март": "03",
        # "април": "04",
        "май": "05",
        "юни": "06",
        "юли": "07",
        # "август": "08",
        # "септември": "09",
        # "октомври": "10",
        # "ноември": "11",
        # "декември": "12",
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


def filter_jobs_by_date(jobs: list, target_date: str = None) -> list:
    """
    Filter jobs by target date
    Args:
        jobs: List of job dictionaries
        target_date: Target date string in 'YYYY-MM-DD' format
    """
    if not target_date:
        target_date = datetime.now().strftime("%Y-%m-%d")

    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    filtered_jobs = []

    for job in jobs:
        try:
            # Get the date text from job
            date_elem = job.find("span", class_="date")
            if not date_elem:
                continue

            job_date = parse_bg_date(date_elem.get_text(strip=True))

            # Compare dates (ignore time)
            if job_date.date() == target_dt.date():
                filtered_jobs.append(job)
        except (ValueError, AttributeError) as e:
            log.error(f"Error parsing date: {e}")
            continue

    return filtered_jobs


def parse_date(date_str):
    """Parse date string and return datetime object"""
    if date_str.lower() == "today":
        return datetime.now()
    elif date_str.lower() == "yesterday":
        return datetime.now() - timedelta(days=1)
    else:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return datetime.now()


def get_category_mapping():
    """Map common category names to dev.bg category parameters"""
    return {
        "data science": "data-science",
        "machine learning": "data-science",
        "data": "data-science",
        "backend development": "back-end-development",
        "python development": "python",
    }


def scrape_dev_bg_jobs(category, target_date):
    """Scrape jobs from dev.bg for a specific category and date"""
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
        for page in range(1, 15):

            base_url = f"https://dev.bg/company/jobs/{category_param}?_paged={page}"

            response = requests.get(base_url, headers=headers, timeout=10)
            # log.info(
            #     f"Fetching jobs for category on page {page}: {category_param} on {target_date.strftime('%Y-%m-%d')}."
            # )
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            job_containers = soup.find_all(
                "div", class_=lambda x: x and x.startswith("job-list-item")
            )

            target_date_str = target_date.strftime("%Y-%m-%d")

            for job in job_containers:
                try:
                    # Extract job information
                    title_elem = job.find(
                        "h6",
                        class_=lambda x: x and "job-title" in x,
                    )
                    title = (
                        title_elem.get_text(strip=True)
                        if title_elem
                        else "Title not found"
                    )

                    # Extract company
                    company_elem = job.find(
                        ["span", "div", "p"],
                        class_=re.compile(r"company|employer", re.I),
                    )
                    company = (
                        company_elem.get_text(strip=True)
                        if company_elem
                        else "Company not specified"
                    )

                    # Extract date
                    date_elem = job.find("span", class_="date")
                    date_posted = (
                        date_elem.get_text(strip=True)
                        if date_elem
                        else "Date not found"
                    )
                    parsed_date = parse_bg_date(date_posted)
                    formatted_date = parsed_date.strftime("%Y-%m-%d")

                    # Extract link
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


def get_todays_jobs(category, date="today"):
    """Get jobs posted today or on a given date in a specific category from dev.bg"""
    try:
        target_date = parse_date(date)
        jobs = scrape_dev_bg_jobs(category, target_date)

        if isinstance(jobs, str):
            return jobs

        if not jobs:
            return f"No jobs found for category '{category}' on {target_date.strftime('%Y-%m-%d')} on dev.bg"

        # Format the response
        result = f"Found {len(jobs)} jobs in '{category}' category for {target_date.strftime('%Y-%m-%d')}:\n\n"

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


if __name__ == "__main__":

    client = OpenAI(
        base_url="http://localhost:8080/v1",
        api_key="not-needed",
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful job search assistant. You can use tools to find job postings on dev.bg.",
        }
    ]

    available_functions = {
        "get_todays_jobs": get_todays_jobs,
    }
    log.info(
        "Job Search Assistant activated. Type 'quit' or 'exit' to end the conversation."
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            log.info("Assistant: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=8192,
        )
        response_message = response.choices[0].message

        while response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)

                log.info(
                    f"Assistant: Thinking... (Calling tool: {function_name} with args: {function_args} )"
                )

                function_response = function_to_call(**function_args)

                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=8192,
            )
            response_message = response.choices[0].message

        final_content = response_message.content
        log.info(f"Assistant: {final_content}")
        messages.append({"role": "assistant", "content": final_content})
