# from agno.agent import Agent
# from agno.models.openai.like import OpenAILike

# MODEL = "mlx-community/Qwen3-8B-8bit"

# if __name__ == "__main__":

#     agent = Agent(
#         model=OpenAILike(
#             id=MODEL,
#             api_key="not-needed",
#             base_url="http://localhost:8080/v1",
#             temperature=0.7,
#             max_tokens=2000,
#             max_completion_tokens=2000,
#         )
#     )

#     agent.print_response("Share a 2 sentence horror story.")

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.tools import tool


@tool(stop_after_tool_call=False)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


agent = Agent(
    instructions="You are a precise engineering assistant.",
    tools=[add],
    model=OpenAILike(
        id="mlx-community/Qwen3-8B-8bit",
        api_key="not-needed",
        base_url="http://localhost:8080/v1",
        temperature=0.2,
        max_tokens=2000,
        max_completion_tokens=2000,
    ),
)

agent.print_response("What is 123.4 + 56.6?")
