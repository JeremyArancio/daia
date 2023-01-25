# [Chain]
chunk_size = 2000
chunk_overlap = 0
separator = "\n"
chain_type = "refine"

# [OpenAI]
embedding_model = "text-embedding-ada-002"
text_model = "text-davinci-002"
temperature = 0.1
max_tokens = 400
top_p = 1
frequency_penalty = 0
presence_penalty = 0
n = 1
stream = False
logprobs = None
verbose = True

# [Prompts]
refine_prompt = """You are a document AI assistant helping the user extract information from a document.
You are given a long document and a question.
Answer the question as if it would come from the document.
Question: {question}
Answer:
"""

template_prompt = """
Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
If you the question is not detailled enough to allow you to find the answer in the document, ask the user to detail more specifically his question.

QUESTION: {question}
=========
"""