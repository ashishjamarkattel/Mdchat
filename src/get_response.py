import os

import openai
from dotenv import load_dotenv

load_dotenv()


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


_PROMPT_TEMPLATE_MARKDOWN = """
    CONTEXT: {context}
    You are a helpful assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Anwser should be formatted in Markdown
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer

    QUESTION: {question}
    ANSWER (formatted in Markdown):
    """

class ResponseLLM:

    def __init__(
            self, 
            context: str, 
            question: str,
            prompt: str = _PROMPT_TEMPLATE,
            prompt_markdown: str = _PROMPT_TEMPLATE_MARKDOWN
            ) :
        

        prompt = prompt.format(
            context=context,
            question=question

        )

        self.prompt_markdown = prompt_markdown.format(
            context=context,
            question=question
        )

        self.knowledge = context
        self.prompt = prompt
        


    
    def _generate(self):
        """Call out to OpenAI's endpoint."""
  
        if len(os.environ["OPENAPI_KEY"])>0:


            openai.api_key = os.environ["OPENAPI_KEY"]
            response = openai.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "user", "content": (self.prompt)},        
                                        ], 
                                        temperature=0.5,
                                        )

        
        return response.choices[0].message.content
    
    @staticmethod
    def _generate_solution(prompt):
        """Call out to OpenAI's endpoint."""
  
        if len(os.environ["OPENAPI_KEY"])>0:


            openai.api_key = os.environ["OPENAPI_KEY"]
            response = openai.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "user", "content": (prompt)},        
                                        ], 
                                        temperature=0.5,
                                        )

        
        return response.choices[0].message.content
    
    def generate_markdown(self):
        """Call out to OpenAI's endpoint."""
  
        if len(os.environ["OPENAPI_KEY"])>0:


            openai.api_key = os.environ["OPENAPI_KEY"]
            response = openai.chat.completions.create(
                                        model="gpt-4o",
                                        messages=[
                                            {"role": "user", "content": (self.prompt_markdown)},        
                                        ], 
                                        temperature=0.5,
                                        )

        
        return response.choices[0].message.content

if __name__=="__main__":

    context = 'ram studies in tai inc.'
    question = "where do ram study?"

    llm = ResponseLLM(
        context=context,
        question=question,
    )
    print(llm._generate())
