from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# define hf llm 
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3",  # or another small model
    task="text-generation",
      
)

# define chat model
chat_model = ChatHuggingFace(llm=llm, verbose=True)


# define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful and experienced educational assistant. "
        "You create high-quality, age-appropriate learning content for students "
        "based on the input provided. Ensure clarity, correct grammar, and relevance to the subject."
    ),
    (
        "user",
        "Create a {length} {tone} {category} in the subject of {subject} for a grade {grade} student. \n"
        "Topic: \"{topic}\"\n\n"
        "Make sure the content is appropriate for the student's level, engaging, and easy to understand. "
        "Use language suitable for {subject} learners in grade {grade}. "
        "Keep the structure and flow organized. Avoid overly complex vocabulary."
    )
])



response = chat_model.invoke(prompt_template.invoke({
    "length": "short",
    "tone": "",
    "category": "Composition",
    "subject": "English",
    "grade": "5",
    "topic": "My Dream Country"
}))
print(response.content)

