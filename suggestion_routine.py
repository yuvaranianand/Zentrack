from dotenv import load_dotenv
import json
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.schema import Document
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os


app = FastAPI()
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

class RecommendationProcessor:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.prompt_template = """ User data: {input_documents}

        Based on the user's data, provide personalized recommendations for self-care and lifestyle improvements. 
        Use the primary_self_care_goal as the main focus for these suggestions. Please provide recommendations 
        in the following JSON format:

        - **routineName**: Provide a brief, one or two-word title for the recommendation.
        - **description**: Give a one-line description of the recommendation tailored to the user's self-care goal.
        - **quotes**: Include a short, motivational quote related to the recommendation.
        - **isAdd**: Set to true to indicate that this recommendation should be added.

        Ensure the recommendations are aligned with the user's self-care goal and presented clearly.
        Try to give at least 15 routines to follow based on the user data.
        Give the proper response in json format.
        """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["input_documents"])
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.chain = StuffDocumentsChain(
            llm_chain=self.llm_chain,
            document_variable_name="input_documents",
        )

    def get_recommendations(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = [Document(page_content=json.dumps(user_data))]
        recommendations = self.chain.invoke(
            {"input_documents": documents}
        )
        response_text = recommendations["output_text"]
        return self.extract_json(response_text)

    @staticmethod
    def extract_json(response_text: str) -> Dict[str, Any]:
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON"}
        else:
            return {"error": "No JSON found in the response"}




class UserInput(BaseModel):
    primary_self_care_goal: str
    dream_to_achieve: str
    difficulty_in_achieving_dreams: str
    satisfying_part_of_day: str
    measures_to_achieve_goals: str
    hours_of_sleep: str
    health_concerns: Optional[str]
    medications: Optional[str]
    habits_disturbing_sleep: str
    frequency_of_stress: str
    self_love: str
    likes_about_self: str
    improvements_in_self: str
    fun_with_everyone: str
    easy_goals: str

def preprocess_user_data(user_input: UserInput) -> Dict[str, Any]:
    processor = RecommendationProcessor()
    return processor.get_recommendations(user_input.dict())

@app.post("/recommendations/")
async def get_recommendations(user_input: UserInput,
                              processor: RecommendationProcessor = Depends(lambda: RecommendationProcessor())):
    user_data = user_input.dict()
    structured_response = processor.get_recommendations(user_data)
    return JSONResponse(content={"message": "Information Extracted successfully", "response": structured_response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
