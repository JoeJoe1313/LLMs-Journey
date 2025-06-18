import json
import re
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

MODEL = "mlx-community/qwen3-4b-bf16"
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
        "backend development": "back-end-development",
        "python development": "python",
    }


def scrape_dev_bg_jobs(category, target_date):
    """Scrape jobs from dev.bg for a specific category and date"""
    try:
        # Map category to dev.bg format
        category_mapping = get_category_mapping()
        category_param = category_mapping.get(
            category.lower(), category.lower().replace(" ", "-")
        )

        # Build URL
        base_url = f"https://dev.bg/company/jobs/{category_param}"

        # Set up headers to mimic a real browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

        # Make request
        response = requests.get(base_url, headers=headers, timeout=10)
        print(
            f"Fetching jobs for category: {category_param} on {target_date.strftime('%Y-%m-%d')}."
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Find job listings - you'll need to inspect dev.bg to get the correct selectors
        job_listings = []

        # Common selectors for job sites (adjust based on dev.bg's actual structure)
        job_containers = soup.find_all(
            ["div", "article"], class_=re.compile(r"job|listing|card", re.I)
        )

        if not job_containers:
            # Try alternative selectors
            job_containers = soup.find_all("a", href=re.compile(r"/job/"))

        target_date_str = target_date.strftime("%Y-%m-%d")

        for job in job_containers[:20]:  # Limit to first 20 jobs
            try:
                # Extract job information
                title_elem = job.find(
                    ["h1", "h2", "h3", "h4"], class_=re.compile(r"title|name|job", re.I)
                )
                if not title_elem:
                    title_elem = job.find("a")

                title = (
                    title_elem.get_text(strip=True) if title_elem else "Title not found"
                )

                # Extract company
                company_elem = job.find(
                    ["span", "div", "p"], class_=re.compile(r"company|employer", re.I)
                )
                company = (
                    company_elem.get_text(strip=True)
                    if company_elem
                    else "Company not specified"
                )

                # Extract date
                date_elem = job.find(
                    ["span", "div", "time"],
                    class_=re.compile(r"date|time|posted", re.I),
                )
                date_posted = (
                    date_elem.get_text(strip=True) if date_elem else "Date not found"
                )

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

                job_listings.append(job_info)

            except Exception as e:
                print(f"Error parsing job: {e}")
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
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
    messages = [
        {
            "role": "user",
            "content": "What are the jobs from today in Data Science category?",
        }
    ]

    functions = {"get_todays_jobs": get_todays_jobs}

    # The first query generates a tool call
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        max_tokens=8192,
    )
    print("First response:", response.choices[0].message)

    # Check if tool calls were made
    if response.choices[0].message.tool_calls:
        # Call the function
        function = response.choices[0].message.tool_calls[0].function
        function_args = json.loads(function.arguments)  # Parse JSON string to dict
        tool_result = functions[function.name](**function_args)

        # Add the tool call to messages
        messages.append(response.choices[0].message)

        # Add the tool result to messages
        messages.append(
            {
                "role": "tool",
                "tool_call_id": response.choices[0].message.tool_calls[0].id,
                "name": function.name,
                "content": tool_result,
            }
        )

        # Generate the final response
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            max_tokens=8192,
        )
        print("\nFinal response:", final_response.choices[0].message.content)
    else:
        print("No tool calls were made")
