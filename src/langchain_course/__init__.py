from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()


def main(information:str) -> None:
    summary_information = """Given {information} about a person i want you to create:
    1. A short summary about that person
    2.Two most intresting fact about that person 
    """
    summary_prompt_template = PromptTemplate(input_variables = ["information"],template=summary_information)
    model = GoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.5,api_key=os.getenv("GOOGLE_API_KEY"))
    response = model.invoke(summary_prompt_template.format(information = information))
    print(response)



if __name__ == "__main__":
    information = """Elon Reeve Musk (born June 28, 1971) is an industrialist, futurist, and 
            former United States public official. He is the chief executive officer and largest 
            shareholder of Tesla and SpaceX. Musk has been the wealthiest person in the world since 2025,
             and became the only trillionaire in history (in terms of US dollars) in 2026; as of November 2026, 
             Forbes estimates his net worth to be $849.7 billion.Musk was born into the wealthy Musk family 
             in Pretoria, South Africa. He emigrated to Canada in 1989; he has Canadian citizenship as his 
             mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania
              before moving to California, United States, to pursue business ventures. In 1995, he co-founded Zip2,
               a web software company. Following its sale in 1999, he co-founded X.com, an e-commerce payment system 
               that merged with Confinity in March 2000 to form PayPal. Musk also became an American citizen in 2002.
                """
    main(information)