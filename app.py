from flask import Flask, request, jsonify, send_file, session
from flask_cors import CORS
import langchain
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import pandas as pd
import json
from fpdf import FPDF
import binascii

secret_key = binascii.hexlify(os.urandom(32)).decode()


load_dotenv()

app = Flask(__name__)
CORS(app)

app.secret_key = secret_key


model = ChatGoogleGenerativeAI(
    model="gemini-pro", google_api_key=os.environ.get("GOOGLE_API_KEY"), temperature=0
)

template="""Answer the question based on the following context:
        {context}
        
        Question : {question}
        """
general_prompt=ChatPromptTemplate.from_template(template)

def get_chunck(document):
    text=""
    page_reader= PdfReader(document)
    for page in page_reader.pages:
        text+=page.extract_text()
    
    if(text.strip()==""):
        return "Empty File"
    else:
        text_spliter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        chunk=text_spliter.split_text(text)
        return chunk

def get_embedding(chuncks):
    if(chuncks=="Empty File"):
        return None 
    else:
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ.get("GOOGLE_API_KEY"))
        vectorstore=FAISS.from_texts(texts=chuncks,embedding=embedding)
        return vectorstore


@app.route("/upload_brd", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        chuncks=get_chunck(file)
        vectorstore=get_embedding(chuncks)
        global retriever
        if vectorstore==None:
            return f'Empty File Uploaded, please try to upload valid BRD.', 406
        else:
            retriever= vectorstore.as_retriever(search_krargs={"k":15})

            template="""Answer the question based on the following context:
            {context}
            
            Question : {question}
            """
            prompt=ChatPromptTemplate.from_template(template)

            chain=( {"context": retriever,"question":RunnablePassthrough()} 
                |prompt | model |StrOutputParser()
                )
            
            file_type=chain.invoke("""[conetext]
            Try to indentify what kind a document's context it was.
            Give me the output as the what kind of document it is. Give me the specific answer.
            If the context is empty or with less content, give the output as "Empty File".
            """)

            if(file_type=="Business Requirement Document" or file_type=="Business Requirements Document (BRD)"):
                return jsonify({"message": "Successfully uploaded BRD file"}), 200
            else:
                print(f"This is a/an {file_type} file, please upload a BRD to generate User story.")
                return jsonify({"error": f"This is a/an {file_type} file"}), 406


@app.route('/get_functionality', methods=['GET'])   
def get_functionality():
    prompt="""Based on this provided [context], please identify the main product [main product] and list of functionality.
    If there any sub functioality on a single functionality, break it down into separate functionality.
    Helpful Answer"""

    chain=({"context": retriever,"question":RunnablePassthrough()} 
          |general_prompt | model |StrOutputParser()
          )
    
    global functionalities
    response=chain.invoke(prompt)
    lines = response.split("\n")
    functionalities = [
        line.strip("* ").split(":")[0].strip()
        for line in lines[1 :]
        if line.startswith("*")
    ]
    # functionalities=model.invoke(f"""Only show the functionality names from the list provided in the response below. The response may vary each time.
    # {response}
    # Show only the fucntionality names from each topics. Combine all functionlity to one.
    # """).content
    print(functionalities)
    return response, 200


@app.route('/get_user_stories', methods=['GET'])
def get_user_story():
    response_schemas = [
    ResponseSchema(name="Title", description="name of the user story"),
    ResponseSchema(
        name="Summary",
        description="A brief, descriptive title for the user story that summarizes its purpose or goal.",
    ),
    ResponseSchema(
        name="Description",
        description="A detailed explanation of the user story, including what needs to be done and why it's important. This should provide enough context for the development team to understand the requirements and objectives.",
    ),
    ResponseSchema(
        name="Acceptance",
        description="Specific conditions or criteria that must be met for the user story to be considered complete. These criteria help define the scope of the work and provide a basis for testing.",
    ),
    ResponseSchema(
        name="Priority",
        description="Indicates the relative importance or urgency of the user story compared to other items in the backlog. This helps the team prioritize their work and focus on the most valuable tasks first.",
    ),
    ResponseSchema(
        name="Story Points",
        description="An estimate of the relative effort or complexity of the user story, often represented using a numerical value such as story points or ideal days. This helps the team understand the size of the task and plan their work accordingly.",
    ),
    ResponseSchema(
        name="Assignee",
        description="The team member responsible for implementing the user story. Assigning tasks to specific team members helps clarify accountability and ensures that work is evenly distributed.",
    ),
    ResponseSchema(
        name="Epic/Theme",
        description="If the user story is part of a larger initiative or project, it may be linked to an epic or theme in JIRA. This provides additional context and helps organize related work items.",
    ),
    ResponseSchema(
        name="Labels/Tags",
        description="Optional labels or tags that provide additional categorization or metadata for the user story. This can be useful for filtering and searching for related items in the backlog.",
    ),
    ResponseSchema(
        name="Linked Issues",
        description="Links to other issues or tasks that are related to the user story, such as dependencies or follow-up work. This helps ensure that the team is aware of any interconnected tasks or requirements.",
    ),
    ]
    
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt_template = """using following
    {text}
    Based on this text, create a user story for the business to upload JIRA.
    When creating user stories in JIRA, it's important to include key attributes to provide clarity and context for the development team. Here are the key attributes typically used when creating user stories in JIRA:

    Note: Create user stories for each functionality as per this name {component_names} and assign the priority as per the priority list {functionality} and also add the priority based on the list.\n This is the format of the user story {format_instructions}.
    
    Helpful Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["text","component_names"],
        partial_variables={"format_instructions": format_instructions,"functionality":functionalities},
    )

    #user_stories_by_component = {}

    user_stories_by_component = []

    chain = LLMChain(llm=model, prompt=prompt, output_parser=output_parser)
    
    # Iterate over each component name
    print(functionalities)
    for component_name in functionalities:
        # Generate user story for the current component
        user_story = chain.run(text=functionalities, component_names=component_name)
        
        # Convert "Labels/Tags" to string if it's a list
        if isinstance(user_story["Labels/Tags"], list):
            user_story["Labels/Tags"] = ', '.join(user_story["Labels/Tags"])
        
        # Convert "Linked Issues" to string if it's a list
        if isinstance(user_story["Linked Issues"], list):
            user_story["Linked Issues"] = ', '.join(user_story["Linked Issues"])

        # Add the generated user story to the dictionary under the component name key
        #user_stories_by_component[component_name] = user_story
        user_stories_by_component.append(user_story)
    
    
    if user_stories_by_component:
        # # Return user stories as JSON response
        return jsonify(user_stories_by_component)
    else:
        return 'No user stories found in session', 404



if __name__ == "__main__":
    app.run(debug=True)