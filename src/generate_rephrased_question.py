import json
from langchain_core.prompts import PromptTemplate
from src.prompts import _standalone_prompt
from src.get_response import ResponseLLM

def paraphrase_question(
        question: str,
):
    prompt = _standalone_prompt.format(
        question=question
    )

    paraphrased_question_json = ResponseLLM._generate_solution(prompt=prompt)   ## generates the multiple paraphrased questions for query.
    
    try:
        p_json = json.loads(paraphrased_question_json)    # read the json file 
    except Exception:
        raise OSError("Cannot read the json file.")
    
    return p_json


if __name__ == "__main__":
    question = "who is cto of tai?"
    print(paraphrase_question(question=question))