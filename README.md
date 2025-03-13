# Simple LangChain Chatbot (with Tools Calling and Memory)

## Performance Review

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e96025ba3e9741a59877775df43355fa)](https://app.codacy.com/gh/D3rhami/Chatbot-LangChain-ToolCalling/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)[![Python Version](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![Latest Version](https://img.shields.io/pypi/v/langchain?label=langchain)](https://pypi.org/project/langchain/)
[![Supported Platforms](https://img.shields.io/badge/platform-linux%20|%20macOS%20|%20windows-green)](https://github.com/your-repo/simple-langchain-chatbot)

---

## Project Overview 🌟

This project demonstrates a **simple yet powerful** use case of **LangChain's tool-calling feature**, enabling seamless
integration of external tools and databases into a conversational AI system. The chatbot is designed to interact with
users and perform key tasks such as:

- **Checking order statuses**
- **Registering new orders**
- **Querying product inventory**

By leveraging **function calling, preprocessing, postprocessing, and memory management**, the system ensures a **dynamic
and user-friendly experience**. This project serves as a practical example of integrating **LangChain** with a backend
system to build an intelligent, task-oriented chatbot for real-world applications.

### **Tech Stack**

- **FastAPI** – Handles the backend API
- **SQLAlchemy** – Manages database interactions
- **LangChain** – Powers the conversational AI logic

This project is an excellent starting point for developers looking to understand how to **build AI-driven chatbots**
with
tool-calling capabilities using LangChain.

## Below is an example of the user interface 🖼️

![User Interface](Screenshot%20ui.png )

---

## Installation and Setup 🛠️

To get started with this project, follow these steps:

1. **Clone the Repository**
   ```bash
   git clone  https://github.com/D3rhami/Chatbot-LangChain-ToolCalling.git
   cd Chatbot-LangChain-ToolCalling
   ```

2. **Install Dependencies**  
   Ensure you have Python 3.8 or higher installed. Then, install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Server**  
   Start the FastAPI server:
   ```bash
   python server.py
   ```

4. **Access the Chatbot**  
   Open your browser and navigate to the user interface:
   ```
   http://127.0.0.1:8000/user.html
   ```
   or just open the user.html file

---

## File Descriptions 📂

<details> <summary>📄 server.py</summary> 
This file sets up the FastAPI server, handles incoming requests, and manages the chatbot's interaction flow. </details>

<details> 
<summary>📄 memory_manager.py</summary> 
This file manages conversation memory for users, ensuring that the chatbot can maintain context across multiple interactions. </details>

<details> 
<summary>📄 function_manager.py</summary> 
This file manages the registration, execution, and response handling of functions that interact with the database and other systems.
</details>

<details>
<summary>📄 preprocess.py</summary> 
This file preprocesses user queries to understand intent and extract entities, preparing the input for function execution.
</details>

<details> 
<summary>📄 process.py</summary>
This file processes the user intent, executes the appropriate function, and generates follow-up questions if needed. </details>

<details> <summary>📄 postprocess.py</summary> 
This file handles the postprocessing of function results, generating human-friendly responses in HTML format. </details>

<details> 
<summary>📄 database.py</summary> 
This file handles all database interactions, including querying the database schema, checking order status, registering new orders, and checking product inventory.
</details>

--- 

## Documentation and Code Quality 📚✨

This project emphasizes clean, well-documented code with detailed comments and structured logic. The documentation,
prompts, and inline comments were generated and refined using an **AI agent** to ensure clarity and maintainability.
Every effort has been made to make the codebase professional, readable, and easy to extend.

---

### Contributions and Feedback 🤝

We welcome contributions, suggestions, and feedback! Feel free to open an issue or submit a pull request to improve this
project. Together, we can make this chatbot even better! 🚀

---

### License 📰

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for more details.

---

Thank you for exploring **Simple LangChain Chatbot (with Tools Calling)**! We hope you find it useful and inspiring for
your own projects. 😊