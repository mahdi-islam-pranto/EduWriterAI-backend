from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# define fastapi app
app = FastAPI()

# define input schema for api
class Item(BaseModel):
    length: str
    word_count: int
    tone: str
    category: str
    subject: str
    grade: str
    topic: str


# base pydantic class for output
class OutputSchema(BaseModel):
    generated_content: str = Field(description="The generated content")
    category: str = Field(description="The category of the generated content")
    tone: str = Field(description="The tone of the generated content")
    word_count: int = Field(description="The total word count of the generated content")



# api endpoint function
@app.post("/generate")
async def root(item: Item):
    try:
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
            "Create a {length} {word_count} {tone} {category} in the subject of {subject} for a grade {grade} student.\n"
            "Topic: \"{topic}\"\n\n"
            "Make sure the content is appropriate for the student's level, engaging, and easy to understand. "
            "Use language suitable for {subject} learners in grade {grade}. "
            "Keep the structure and flow organized. Avoid overly complex vocabulary."
            )
        ])

        # convert pydantic class to json schema
        output_json_schema = OutputSchema.model_json_schema()
        # define response with structure
        response_with_structure = chat_model.with_structured_output(output_json_schema)
        # get AI response
        AI_Response = response_with_structure.invoke(prompt_template.invoke({
            "length": item.length,
            "word_count": item.word_count,
            "tone": item.tone,
            "category": item.category,
            "subject": item.subject,
            "grade": item.grade,
            "topic": item.topic
        }))

        # return response
        return {
            "status": "success",
            "AI_Response": AI_Response,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

