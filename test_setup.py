import os  

from dotenv import load_dotenv 
from langchain_openai import ChatOpenAI   

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")  

if not api_key: 
    raise ValueError( 
    "OPENROUTER_API_KEY was not found. " 
    "Confirm that your .env file is in the same folder " 
    "and that the variable name is correct."
    )  

llm = ChatOpenAI( 
model="openai/gpt-4o-mini", 
base_url="https://openrouter.ai/api/v1",
api_key=api_key, 
temperature=0, 
max_tokens=50,
)  

try: 
    response = llm.invoke("Say hello in one short sentence.") 
    print(response.content) 
    print("\nSetup verification completed successfully.")  

except Exception as error: 
    print("The setup verification request failed.") 
    print(f"Error type: {type(error).__name__}") 
    print(f"Details: {error}")
