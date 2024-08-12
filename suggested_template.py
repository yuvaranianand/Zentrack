from dotenv import load_dotenv
import json
import csv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.schema import Document
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import os
import re

app = FastAPI()
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')


class RecommendationProcessor:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.prompt_template = """ User data: {input_documents}

        Based on the provided routine name, generate a detailed recommendation including:
        - **routineName**: The name of the routine.
        - **description**: A detailed description of the recommendation.
        - **quote**: A motivational quote related to the routine.

        Also, determine the category for the routine from the following options:
        ['wake up', 'song time', 'workout', 'yoga and meditation', 'positive affirmations', 'breakfast', 
        'drink water', 'getting ready for office', 'cooking', 'upskilling', 'reading time', 'lunch', 
        'household work', 'making bed', 'dinner', 'prepare for next day', 'thanksgiving before bed', 
        'planned schedules', 'a date', 'movie night', 'self care', 'family time', 'medication', 'nan', 
        'morning study', 'getting ready', 'fresh up and snack', 'study time', 'playtime', 'special classes'].

        Provide the response in JSON format including the image URL corresponding to the category.
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

    def get_recommendations(self, routine_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = [Document(page_content=json.dumps(routine_data))]
        recommendations = self.chain.invoke({"input_documents": documents})
        response_text = recommendations.get("output_text", "")
        structured_response = self.extract_json(response_text)

        
        routine_name = routine_data.get("routineName", "").strip()
        category = structured_response.get("category", "").strip().lower() 

        if category in self.category_image_urls:
            image_url = self.category_image_urls[category]
            structured_response["imageUrl"] = image_url
        else:
            structured_response["imageUrl"] = None

        
        structured_response.pop("category", None)

        return structured_response

    @staticmethod
    def extract_json(response_text: str) -> Dict[str, Any]:
        try:
            
            response_text = response_text.strip()
            if response_text.startswith("{") and response_text.endswith("}"):
                
                return json.loads(response_text)
            else:
              
                json_str = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_str:
                    return json.loads(json_str.group(0))
                return {"error": "Response does not look like valid JSON"}
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON: {str(e)}"}


class RoutineInput(BaseModel):
    routineName: str


@app.post("/template/")
async def get_recommendations(routine_input: RoutineInput,
                              processor: RecommendationProcessor = Depends(lambda: RecommendationProcessor())):
    routine_data = routine_input.dict()
    structured_response = processor.get_recommendations(routine_data)
    return JSONResponse(content={"message": "Information Extracted successfully", "response": structured_response})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
