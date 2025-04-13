from dotenv import load_dotenv
import streamlit as stream
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate, 
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables import RunnableLambda

load_dotenv('./../.env')

# To set the background proportion to wide
stream.set_page_config(layout="wide")

# Page title and description
stream.title("Online Tutoring")
stream.write("Chat with a Adaptive AI Tutor")

# Base settings for the chatbot
base_url = "http://localhost:11434"
model = 'llama3.2:latest'
user_id = stream.text_input("Enter the user ID", "Shreyash Raj Bamrara")

# Function to get chat history from the file chat_history.db
def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

# Initialize session state for chat history
if "chat_history" not in stream.session_state:
    stream.session_state.chat_history = []

# Button to start a new conversation
if stream.button("Start New Conversation"):
    stream.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

for message in stream.session_state.chat_history:
    with stream.chat_message(message['role']):
        stream.markdown(message['content'])

# LLM setup
llm = ChatOllama(base_url=base_url, model=model)

# Prompt templates
about_faculty_selector = ''' 
You are a classification bot designed to analyze a question and classify it into one of five predefined categories. 
Your task is to respond with exactly one word, corresponding to the most appropriate category based solely on the given input question. 
You must strictly follow these rules:

1. You are only allowed to respond with one of the following words: "dsa", "careerskill", "maths", "oops", or "default".
2. Do not provide explanations, clarifications, or any text other than the one-word response.
3. The response must match the input question to the categories below:
   - "dsa": For questions related to data structures.
   - "careerskill": For questions related to career skills, English grammar, literature, logical reasoning, or if any of these terms or topics are mentioned.
   - "maths": For questions about mathematics or if terms like "math," "algebra," "set theory," "graph theory," or "combinatorics" are present.
   - "oops": For questions on object-oriented programming, or if "OOP" or "object-oriented" is explicitly mentioned.
   - "default": For any other input where none of the above categories apply or the input is unclear.
4. Do not consider previous context or history. Base your decision solely on the content of the current input question.
5. If multiple categories could apply, select the most specific match based on the keywords provided above.
6. If the input contains common greetings (e.g., "hello", "good morning", "good evening") or name introductions, respond with "default". These are not to be classified into any of the predefined categories.
7. If the input seems like a general conversation, introductory phrase, or includes a name, return "default" to avoid misclassification into specific categories.

Your response should be concise, accurate, and strictly adhere to the rules. Respond now based on the given input question.
'''

# Faculty prompts
about_cc = '''
You are Professor Gayu from Graphic Era Hill University, Class Coordinator of Section A1. Known for your charismatic charm and heartwarming demeanor, you approach students with a mix of professional rigor and personal care. Just like Shah Rukh Khan, you have a way with words, often inspiring your students with profound thoughts. 
Your catchphrase: *"Success is not a destination, its the journey you take with courage and determination."*
You often use dramatic analogies and a hint of humor to keep the class engaged, making even tough academic topics feel like a cinematic experience.
'''

about_ds = '''
You are Professor Magma, a no-nonsense expert in Data Structures with C. Like Gordon Ramsay, you demand excellence and have zero tolerance for mediocrity. While you may be tough on your students, your passion for perfection pushes them to deliver their best.
Your catchphrase: *"This is not just code—its a masterpiece or nothing at all, you idiot sandwich!"*
You emphasize rigorous implementation of algorithms and direct your students with an unfiltered, straightforward approach, ensuring they grasp the depth of the subject.
'''

about_career_skill = '''
You are Professor Jena, a compassionate and optimistic expert in English grammar and logical reasoning. Much like Shashi Tharoor, your eloquence, expansive vocabulary, and witty use of language captivate your students. Your intellectual brilliance is balanced by your kindness, allowing students to grow through their mistakes.
Your catchphrase: *"Remember, words are not just tools of communication; they are the symphony of thought."*
You make concepts relatable through sophisticated analogies, while encouraging your students to embrace the art of articulation and reasoning.
'''

about_oops = '''
You are Professor Karma, an experienced expert in Object-Oriented Programming. Like Thanos, you believe in balance, discipline, and achieving goals through hard work. While you are strict and direct, you occasionally inject humor and captivating analogies.
Your catchphrase: *"Reality can be whatever you make it—with enough practice and research."*
For every question, you assign five research topics to expand your students' horizons, promoting a universe of endless learning possibilities.
'''

about_maths = '''
You are Professor Zenith, an authoritative and motivational expert in Discrete Mathematics. Like Narendra Modi, your teaching style is structured and inspiring, often using real-world examples and a logical approach to engage your students. You maintain a disciplined demeanor but are approachable to those who show genuine effort.
Your catchphrase: *"Small efforts lead to big changes—one step at a time in mathematics, one proof at a time."*
You simplify complex topics through step-by-step problem breakdowns, ensuring that your students develop a robust and practical understanding of mathematics.
'''

# Templates for each faculty chain with dynamic user input
cc_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(about_cc),
     MessagesPlaceholder(variable_name='history'),  
    HumanMessagePromptTemplate.from_template("{input}")   
])

dsa_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(about_ds),
     MessagesPlaceholder(variable_name='history'),
    HumanMessagePromptTemplate.from_template("{input}")
])

maths_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(about_maths),
    HumanMessagePromptTemplate.from_template("{input}")
])

oops_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(about_oops),
     MessagesPlaceholder(variable_name='history'),
    HumanMessagePromptTemplate.from_template("{input}")
])

career_skill_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(about_career_skill),
     MessagesPlaceholder(variable_name='history'),
    HumanMessagePromptTemplate.from_template("{input}")
])

# Faculty-specific chains
cc_chain = cc_template | llm | StrOutputParser()
dsa_chain = dsa_template | llm | StrOutputParser()
maths_chain = maths_template | llm | StrOutputParser()
oops_chain = oops_template | llm | StrOutputParser()
career_skill_chain = career_skill_template | llm | StrOutputParser()

# Selector chain setup
selector_system = SystemMessagePromptTemplate.from_template(about_faculty_selector)
selector_messages = [selector_system, HumanMessagePromptTemplate.from_template("{input}")]
selector_prompt = ChatPromptTemplate(messages=selector_messages)
selector_chain = selector_prompt | llm | StrOutputParser()

# Routing function
def rout(info):
    subject = info["subject"].lower()
    if 'dsa' in subject:
        return dsa_chain
    elif 'maths' in subject:
        return maths_chain
    elif 'careerskill' in subject:
        return career_skill_chain
    elif 'oops' in subject:
        return oops_chain
    else:
        return cc_chain

required_chain = {"subject": selector_chain, 'input': lambda x: x['input']} | RunnableLambda(rout)

runnable_with_history = RunnableWithMessageHistory(required_chain, get_session_history,
                                                   input_messages_key='input', history_messages_key='history')

# Chatting function
def chat_with_llm(session_id, input):
    for output in runnable_with_history.stream({'input': input}, config={'configurable': {'session_id': session_id}}):
        yield output

# Chat input interface for the user
# Conversation box given by Streamlit
prompt = stream.chat_input("Send a Message to our Chatbot")

if prompt:
    # Appending the user message to chat history
    stream.session_state.chat_history.append({'role': 'user', 'content': prompt})
    with stream.chat_message("user"):
        stream.markdown(prompt)

    #Using the selector chain to classify the subject
    classification = selector_chain.invoke({'input': prompt, 'history': stream.session_state.chat_history})
    subject = classification.strip().lower()

    # Route to the correct chain using the subject classification
    chain = rout({"subject": subject, "input": prompt})

    #Passing the user's input and history to the chain
    response = chain.invoke({'input': prompt, 'history': stream.session_state.chat_history})

    # Step 4: Append the assistant's response to the chat history
    stream.session_state.chat_history.append({'role': 'assistant', 'content': response})
    with stream.chat_message("assistant"):
        stream.markdown(response)

