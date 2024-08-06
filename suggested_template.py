from dotenv import load_dotenv
import json
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
        - **articleUrl**: A URL link to a relevant article or resource related to the routine. Give proper related link
        should extract from this.

        Please ensure the recommendation is aligned with the routine name and is presented clearly.
        Provide the response in JSON format.
        """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["input_documents"])
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.chain = StuffDocumentsChain(
            llm_chain=self.llm_chain,
            document_variable_name="input_documents",
        )

    def get_recommendations(self, routine_data: Dict[str, Any]) -> Dict[str, Any]:
        documents = [Document(page_content=json.dumps(routine_data))]
        recommendations = self.chain.invoke({"input_documents": documents})
        response_text = recommendations.get("output_text", "")
        #print(f"Full recommendations response: {recommendations}")  # Debugging line
        #print(f"Raw response text: {response_text}")  # Debugging line
        return self.extract_json(response_text)

    @staticmethod
    def extract_json(response_text: str) -> Dict[str, Any]:
        #print(f"Raw response text: {response_text}")  # Debugging line
        try:
            # Strip any extraneous characters or text
            response_text = response_text.strip()
            if response_text.startswith("{") and response_text.endswith("}"):
                # Directly try to parse JSON
                return json.loads(response_text)
            else:
                # Handle cases where the response might be wrapped in extra text
                # This is a fallback; adjust if you can identify specific patterns
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
