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
        "based on the input provided. Ensure clarity, correct grammar, and relevance to the subject.\n\n"
        "The learning materials you create must fall into one of the following categories:\n\n"
        "1. Composition - A structured, multi-paragraph piece of writing expressing thoughts or opinions on a topic.\n"
        "2. Paragraph - A single, concise paragraph describing a topic clearly and simply.\n"
        "3. Poem - A short, creative, and rhythmic piece suitable for children of the specified grade.\n"
        "4. MCQ - A set of multiple-choice questions with 4 options each and one correct answer.\n"
        "5. Summary - A brief overview capturing the key ideas of a given topic or passage.\n"
        "6. Explanation - A clear and simple explanation of a topic or concept.\n"
        "7. Dialogue - A short scripted conversation between two or more characters, relevant to the topic.\n"
        "8. Essay - A longer, more in-depth piece of writing that explores a topic in detail with multiple paragraphs (with clear points and titles) and supporting details.\n"
        "9. Email - A simulated email style writing with a well email format to a person, relevant to the topic.\n"
        "10. Story - A short, imaginative piece of writing that tells a story, relevant to the topic.\n"
        "11. Letter - A simulated letter style writing to a person, relevant to the topic.\n"
        "12. Application - A simulated application style writing to a person, relevant to the topic.\n\n"
        "Always follow the structure and tone appropriate for the given category and grade level."
    ),
    (
        "user",
        "Create a {length} {tone} {category} in the subject of {subject} for a grade {grade} student.\n"
        "Topic: \"{topic}\"\n\n"
        "Make sure the content is appropriate for the student's level, engaging, and easy to understand. "
        "Use language suitable for {subject} learners in grade {grade}. "
        "Keep the structure and flow organized. Avoid overly complex vocabulary."
    )
])


response = chat_model.invoke(prompt_template.invoke({
    "length": "short",
    "tone": "academic",
    "category": "Composition",
    "subject": "English",
    "grade": "10",
    "topic": "A School Library"
}))
print(response.content)

