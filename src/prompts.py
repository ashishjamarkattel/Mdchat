_standalone_prompt = """Given the following query 
###Question:
{question}
rephrase the question to be a standalone question, in its original language, 
that can be used to query a FAISS index. This query will be used to retrieve documents with additional context.
Give be the 5 rephrased question, format your answer in json object like.
'
  "0":"rephrased question 0",
  "1":"rephrased question 1",
  "2":"rephrased question 2",
  "3":"rephrased question 3",
  "4":"rephrased question 4",
'
"""