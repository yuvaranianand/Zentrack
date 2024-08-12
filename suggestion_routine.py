from dotenv import load_dotenv
import json
import re
import csv
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
        Each recommendation should include:
        - **routineName**: Provide a brief, one or two-word title for the recommendation.
        - **description**: Give a one-line description of the recommendation tailored to the user's self-care goal.
        - **quotes**: Include a short, motivational quote related to the recommendation.
        - **category**: Determine a category for the routine from the following options: 
          ['wake up', 'song time', 'workout', 'yoga and meditation', 'positive affirmations', 'breakfast', 
          'drink water', 'getting ready for office', 'cooking', 'upskilling', 'reading time', 'lunch', 
          'household work', 'making bed', 'dinner', 'prepare for next day', 'thanksgiving before bed', 
          'planned schedules', 'a date', 'movie night', 'self care', 'family time', 'medication', 'nan', 
          'morning study', 'getting ready', 'fresh up and snack', 'study time', 'playtime', 'special classes'].
        - **isAdd**: Set to true to indicate that this recommendation should be added.

        Provide the response in JSON format including the category and an image URL corresponding to the category. 
        Give at least 10 to 15 recommendations
        """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["input_documents"])
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.chain = StuffDocumentsChain(
            llm_chain=self.llm_chain,
            document_variable_name="input_documents",
        )
        self.category_image_urls = self.load_category_image_urls("routine.csv")

    def load_category_image_urls(self, csv_file_path: str) -> Dict[str, str]:
        category_image_urls = {}
        try:
            with open(csv_file_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    category = row['NAME'].strip().lower()  # Normalize category names
                    image_url = row['IMAGE URL'].strip()
                    category_image_urls[category] = image_url
        except Exception as e:
            print(f"Error loading CSV file: {e}")
        return category_image_urls

    def get_recommendations(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = [Document(page_content=json.dumps(user_data))]
        recommendations = self.chain.invoke({"input_documents": documents})
        response_text = recommendations["output_text"]
        routines = self.extract_json(response_text)

        if isinstance(routines, list):
            for routine in routines:
                category = routine.get("category", "").strip().lower()  # Normalize category names
                routine["imageUrl"] = self.category_image_urls.get(category, None)
            # Exclude category field from the final output
            for routine in routines:
                if "category" in routine:
                    del routine["category"]
        else:
            return {"error": "Unexpected response format"}

        return routines

    @staticmethod
    def extract_json(response_text: str) -> Any:
        try:
            if response_text.startswith("[") and response_text.endswith("]"):
                return json.loads(response_text)
            else:
                json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(0))
                return {"error": "No JSON found in the response"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON"}

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

@app.post("/recommendations/")
async def get_recommendations(user_input: UserInput,
                              processor: RecommendationProcessor = Depends(lambda: RecommendationProcessor())):
    user_data = user_input.dict()
    structured_response = processor.get_recommendations(user_data)
    return JSONResponse(content={"message": "Information Extracted successfully", "response": structured_response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
