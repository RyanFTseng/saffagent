from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer
import os
from dotenv import load_dotenv

# Debug: Check current directory and .env file
print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

# Check if .env exists
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
print(f"Looking for .env at: {env_path}")
print(f".env exists: {os.path.exists(env_path)}")


load_dotenv()

judgment_key = os.getenv("JUDGMENT_API_KEY")
judgment_org_id = os.getenv("JUDGMENT_ORG_ID")

client = JudgmentClient(
    judgment_key,
    judgment_org_id
)

example = Example(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."],
)

scorer = FaithfulnessScorer(threshold=0.5)
results = client.run_evaluation(
    examples=[example],
    scorers=[scorer],
    model="gpt-4o",
    project_name="your-project-name"
)
print(results)