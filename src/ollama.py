import requests
import json

_PROMPT_TEMPLATE = """Please Behave as Customer Support. Your task is to provide answers to the given questions based on the provided context. 
Only use your knowledge to craft the answer from the given context, add details if you know any and make it clear and comprehensive.
If you need full context to answer some questions, respectfully respond with that but don't try to give incomplete answer.

If you cannot find the answer in the given context, Please respond with you are not aware about the question and ask for context.
 Don't try to makeup the answer.
    

Context: {context}

Customer Question is listed in triple backticks.

```{question}```

Your Helpful Answer:

"""

_URL = "http://localhost:11434/api/chat"

class OllamaGeneration:
    def __init__(
            self, 
            model: str,
            context: str, 
            question: str,
            prompt: str = _PROMPT_TEMPLATE, 
            url: str = _URL,  
            ) :
        

        prompt = prompt.format(
            context=context,
            question=question

        )
        self.model = model
        self.knowledge = context
        self.prompt = prompt
        self.url = url
    
    def _generate(self):
        """Call out to any model that is selected."""
        data = {
            "model": self.model, 
            "messages": [
                {
                    "role": "user", 
                    "content": self.prompt,
                    "stream": False
                }
            ]
        }
        response = requests.post(self.url, json=data)
        if response.status_code == 200:
            response_text = response.text
            response_lines = response_text.splitlines()
            response_json = [json.loads(line) for line in response_lines]
            text = ""
            for line in response_json:
                text = f"{text}{line['message']['content']}"
            return text
        else:
            return f"Got a different response: {response.status_code}"







