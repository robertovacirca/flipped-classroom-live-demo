import streamlit as st
import streamlit_authenticator as stauth # Ensure this is installed: pip install streamlit-authenticator
import os
import shutil
import logging
from pathlib import Path
import yaml
from yaml.loader import SafeLoader

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    Document,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Financial Report Chatbot", page_icon="💰", layout="wide")

# --- Configuration Constants ---
STORAGE_DIR = "./storage"
UPLOAD_DIR = "./uploads"
OPENAI_MODEL_NAME = "gpt-4.1-mini" # OpenAI Default Model
EMBEDDING_MODEL_NAME = "text-embedding-3-small" # Default Embedding Model
TEMPERATURE = 0.3
TOP_K_RETRIEVAL = 5
SYSTEM_PROMPT = """
You are an expert financial analyst AI assistant.
You are interacting with a user who has uploaded one or more financial documents.
Carefully analyze the user's questions and the relevant context from the documents.
Provide clear, concise, and accurate answers based *only* on the information present in the provided documents.
If the information is not available in the documents, state that clearly.
Do not make up information. You can provide external knowledge only if it context is clear.
Always use markdown for the responses. You can use, if needed, tables to display financial information.
Do not use latex formula.
"""

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def list_indexes():
    """Lists the available indexes in the storage directory."""
    if not os.path.exists(STORAGE_DIR):
        return []
    return [d for d in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, d))]

def clear_directory(dir_path):
    """Removes all files and subdirectories within a given directory."""
    if not os.path.exists(dir_path):
        return
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            st.error(f"Failed to delete {item_path}. Reason: {e}")

@st.cache_resource(ttl="1h")
def load_index(index_name: str, _llm, _embed_model) -> VectorStoreIndex | None:
    """Loads an existing index from the storage directory, ensuring Settings are updated."""
    index_path = os.path.join(STORAGE_DIR, index_name)
    if not os.path.exists(index_path):
        st.error(f"Index '{index_name}' not found.")
        return None
    try:
        logging.info(f"Loading index '{index_name}' from {index_path} with LLM: {getattr(_llm, 'model', 'N/A')} and Embed: {getattr(_embed_model, 'model_name', 'N/A')}")
        # Temporarily set global Settings for loading, as load_index_from_storage might use them.
        # The passed _llm and _embed_model are crucial for cache invalidation.
        original_llm, original_embed_model = Settings.llm, Settings.embed_model
        Settings.llm = _llm
        Settings.embed_model = _embed_model
        
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        
        # Restore original settings if necessary, though the app flow usually re-sets them globally.
        Settings.llm, Settings.embed_model = original_llm, original_embed_model
        
        logging.info(f"Index '{index_name}' loaded successfully.")
        return index
    except Exception as e:
        st.error(f"Error loading index '{index_name}': {e}")
        logging.error(f"Error loading index '{index_name}': {e}", exc_info=True)
        return None

@st.cache_resource(ttl="1h")
def get_chat_engine(_index: VectorStoreIndex): # _index is the key for caching
    """Gets a chat engine from the loaded index, using current global Settings.llm."""
    logging.info(f"Creating chat engine for index object ID: {id(_index)}...")
    desired_top_k = TOP_K_RETRIEVAL
    logging.info(f"Configuring chat engine with similarity_top_k={desired_top_k}")

    # The chat engine will use the LLM currently configured in global Settings.llm
    chat_engine = _index.as_chat_engine(
        similarity_top_k=desired_top_k,
        chat_mode="condense_plus_context",
        llm=Settings.llm, # Explicitly use the globally configured LLM
        context_prompt=(
            "You are a helpful financial analyst AI assistant.\n"
            "Here are the relevant documents for the user's query:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Based *only* on the provided documents, please answer the query.\n"
            "If the answer is not in the documents, state that clearly.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        system_prompt=SYSTEM_PROMPT,
        streaming=True,
        verbose=True # Set to False for less console output in production
    )
    llm_identifier = getattr(Settings.llm, 'model', getattr(Settings.llm, 'model_name', 'Unknown LLM Type'))
    logging.info(f"Chat engine created using LLM identifier: {llm_identifier}")
    return chat_engine

# --- Authentication ---
# Ensure you have a 'config.yaml' file in the same directory with credentials
# Example config.yaml:
# credentials:
#   usernames:
#     jsmith:
#       email: jsmith@example.com
#       name: John Smith
#       password: 'hashed_password_for_abc' # Use stauth.Hasher(['your_password']).generate() to get this
#     rbriggs:
#       email: rbriggs@example.com
#       name: Rebecca Briggs
#       password: 'hashed_password_for_def'
# cookie:
#   expiry_days: 30
#   key: 'some_signature_key' # Must be string
#   name: 'some_cookie_name'
# preauthorized:
#   emails:
#   - melsby@example.com

import streamlit as st
import logging
import yaml
import streamlit_authenticator as stauth
from yaml import SafeLoader

# Try accessing the secrets from `secrets.toml`
import streamlit as st
import logging
import streamlit_authenticator as stauth

# Try accessing the secrets from `secrets.toml`
try:
    # Check if 'credentials' and 'cookie' sections are available in secrets
    if 'credentials' not in st.secrets or 'cookie' not in st.secrets:
        st.error("Error: 'secrets.toml' is missing 'credentials' or 'cookie' section.")
        st.stop()

    # Extract user data and cookie data from secrets.toml
    credentials = st.secrets['credentials']
    cookie = st.secrets['cookie']

    # The `credentials` section in secrets.toml should already have usernames as keys
    # and user details as nested dictionaries (e.g., 'professor' => {'email': '...', 'first_name': '...', 'password': '...'}).
    # We need to convert this into the format expected by `stauth.Authenticate`.

    usernames = list(credentials.keys())  # List of usernames (e.g., ['professor'])
    passwords = [user['password'] for user in credentials.values()]  # List of passwords (hashed or plain)
    full_names = [f"{user['first_name']} {user['last_name']}" for user in credentials.values()]  # Full names of users
    emails = [user['email'] for user in credentials.values()]  # Emails of users

    # Construct the credentials dictionary with required fields in the correct format
    credentials_dict = {
        'usernames': {username: {  # Each username is a key with associated user data as a dictionary
            'email': user['email'],
            'first_name': user['first_name'],
            'last_name': user['last_name'],
            'password': user['password'],
            'failed_login_attempts': user.get('failed_login_attempts', 0),
            'logged_in': user.get('logged_in', False)
        } for username, user in credentials.items()},
    }

    # Initialize the authenticator with the credentials and cookie data
    authenticator = stauth.Authenticate(
        credentials_dict,  # Pass the entire credentials dictionary
        cookie['name'],  # Cookie name
        cookie['key'],   # Cookie key
        cookie['expiry_days']  # Expiry days for the cookie
    )
except KeyError as e:
    st.error(f"Authentication Error: Missing key in 'secrets.toml': {e}")
    st.stop()
except Exception as e:
    st.error(f"Authentication Error: Failed to initialize authenticator: {e}")
    logging.error(f"Authenticator initialization error: {e}", exc_info=True)
    st.stop()

# Render login module
# name, authentication_status, username = authenticator.login('Login', 'main')
# The login widget can be placed in 'main' or 'sidebar'.
# Let's use a more common pattern where login() is called and then content is gated.
#name, authentication_status, username = authenticator.login()
try:
    authenticator.login()
except Exception as e:
    st.error(e)

authentication_status = st.session_state.get('authentication_status')
name = st.session_state.get('name')

if authentication_status:
    # --- User is Authenticated - Main Application Logic ---
    with st.sidebar:
        st.write(f'Welcome *{name}*')
        authenticator.logout('Logout', 'sidebar') # Logout button in the sidebar
        st.divider()
        st.header("Configuration")

        # --- Model Selection ---
        st.subheader("Chat Model")
        model_options = {
            f"OpenAI ({OPENAI_MODEL_NAME})": "openai",
            # Add other model options here if needed
        }
        # Initialize session state for model selection if not already present
        if "selected_model_option" not in st.session_state:
            st.session_state.selected_model_option = "openai" # Default to OpenAI

        selected_model_key = st.radio(
            "Choose the chat model:",
            options=list(model_options.keys()), # Ensure options is a list
            key="model_choice_radio",
            index=list(model_options.values()).index(st.session_state.selected_model_option)
        )
        selected_model_type = model_options[selected_model_key]
        st.caption(f"Using {selected_model_key} for chat.")
        st.divider()

        # --- Index Management ---
        st.header("Index Management")
        Path(STORAGE_DIR).mkdir(parents=True, exist_ok=True)
        Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

        st.subheader("Create New Index")
        index_name_input = st.text_input("Enter a name for the new index:", key="index_name_input")
        uploaded_files = st.file_uploader(
            "Upload financial documents (PDF, DOCX, TXT)",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt']
        )

        if st.button("Create Index", key="create_index_button"):
            if not index_name_input:
                st.warning("Please enter a name for the index.")
            elif not uploaded_files:
                st.warning("Please upload at least one document.")
            else:
                index_path = os.path.join(STORAGE_DIR, index_name_input)
                if os.path.exists(index_path):
                    st.error(f"Index name '{index_name_input}' already exists. Choose a different name.")
                else:
                    with st.spinner(f"Creating index '{index_name_input}'... This may take a moment."):
                        try:
                            clear_directory(UPLOAD_DIR) # Clear previous uploads
                            temp_doc_paths = []
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                temp_doc_paths.append(file_path)
                            logging.info(f"Saved uploaded files: {temp_doc_paths}")

                            # Load documents (ensure Settings.embed_model is set before this)
                            if not Settings.embed_model: # Should be set globally before this point
                                st.error("Embedding model not configured. Cannot create index.")
                                raise ValueError("Embedding model not configured for index creation.")

                            documents = []
                            # SimpleDirectoryReader needs a directory, not a list of files for input_files in this context
                            # Better to process one by one or ensure UPLOAD_DIR contains only current files
                            for file_path_obj in Path(UPLOAD_DIR).glob('*.*'): # Read all files from UPLOAD_DIR
                                reader = SimpleDirectoryReader(input_files=[str(file_path_obj)])
                                docs_for_file = reader.load_data(show_progress=True)
                                for doc in docs_for_file:
                                    doc.metadata["filename"] = file_path_obj.name
                                documents.extend(docs_for_file)
                            logging.info(f"Loaded {len(documents)} total documents for indexing from {UPLOAD_DIR}.")

                            if not documents:
                                st.error("Could not load any documents from the uploaded files.")
                            else:
                                # VectorStoreIndex.from_documents uses global Settings.embed_model
                                index = VectorStoreIndex.from_documents(
                                    documents, show_progress=True
                                )
                                logging.info(f"Persisting index '{index_name_input}' to {index_path}")
                                index.storage_context.persist(persist_dir=index_path)
                                st.success(f"Index '{index_name_input}' created successfully!")
                                st.rerun() # Rerun to update index list and clear inputs
                        except Exception as e:
                            st.error(f"Error creating index: {e}")
                            logging.error(f"Error creating index '{index_name_input}': {e}", exc_info=True)
                        finally:
                            clear_directory(UPLOAD_DIR) # Clean up
                            logging.info(f"Cleaned up temporary directory: {UPLOAD_DIR}")
        
        st.divider()
        st.subheader("Select Index for Chat")
        available_indexes = list_indexes()
        selected_index_name = None # Initialize

        if not available_indexes:
            st.info("No indexes created yet. Upload documents and create an index to start chatting.")
        else:
            selected_index_name = st.selectbox(
                "Choose an index:",
                options=available_indexes,
                key="selected_index",
                index=None, # No default selection
                placeholder="Select an index..."
            )

    # --- Main App Area (Title, Chat, etc.) ---
    st.title("💬 Financial Report Chatbot")
    st.caption(f"Chat with your financial documents using LlamaIndex and Streamlit.")

    # --- API Key and Global Settings Configuration ---
    openai_api_key = st.secrets.get("OPENAI_API_KEY")

    # Configure Embedding Model (globally)
    try:
        if not openai_api_key and selected_model_type == "openai": # Assuming embeddings also need OpenAI key
             st.warning("OpenAI API key not found in secrets. OpenAI Embeddings may fail.", icon="🔑")
        # This should ideally be set once, or be robust to re-setting
        if not hasattr(Settings, 'embed_model_configured') or not Settings.embed_model_configured:
            Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL_NAME, api_key=openai_api_key)
            Settings.embed_model_configured = True # Flag to avoid re-init unless necessary
            logging.info(f"Global OpenAI Embedding model configured: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        st.error(f"Error initializing OpenAI Embedding model: {e}")
        Settings.embed_model = None # Ensure it's None if failed

    # Configure LLM (globally) based on Sidebar Selection
    llm_instance = None
    try:
        if selected_model_type == "openai":
            if not openai_api_key:
                st.warning("OpenAI API key not found in secrets. OpenAI model may not work.", icon="🔑")
            llm_instance = OpenAI(
                model=OPENAI_MODEL_NAME,
                temperature=TEMPERATURE,
                api_key=openai_api_key
            )
            logging.info(f"Configured to use OpenAI LLM: {OPENAI_MODEL_NAME}")
        # Add other model types here (e.g., else if selected_model_type == "lmstudio": ...)

        if llm_instance:
            Settings.llm = llm_instance # Update global LLM setting
        else:
            # This case implies selected_model_type was not handled or failed before llm_instance assignment
            if selected_model_type: # Only error if a selection was made but not handled
                st.error(f"Failed to configure the selected LLM: {selected_model_key}")
            Settings.llm = None
    except Exception as e:
        st.error(f"Error initializing selected LLM ({selected_model_key}): {e}")
        logging.error(f"Error initializing LLM (Type: {selected_model_type}): {e}", exc_info=True)
        Settings.llm = None # Ensure Settings.llm is None on error

    # --- Detect Model Change and Clear Engine/Messages ---
    if st.session_state.selected_model_option != selected_model_type:
        logging.info(f"Model selection changed from {st.session_state.selected_model_option} to {selected_model_type}. Clearing chat engine and messages.")
        st.session_state.chat_engine = None
        st.session_state.messages = []
    st.session_state.selected_model_option = selected_model_type # Update the state

    # --- Initialize/Reset Session State Variables for Chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    # Tracks the index name for which the current engine was built
    if "current_engine_index" not in st.session_state:
        st.session_state.current_engine_index = None

    # --- Load Index and Prepare Chat Engine ---
    # Only proceed if an index is selected AND the LLM is configured
    if selected_index_name and Settings.llm and Settings.embed_model:
        # If engine is not set, or if the selected index has changed, or if model changed (handled above by clearing engine)
        if st.session_state.chat_engine is None or st.session_state.current_engine_index != selected_index_name:
            llm_display_name = getattr(Settings.llm, 'model', selected_model_key)
            with st.spinner(f"Loading index '{selected_index_name}' and preparing chat with {llm_display_name}..."):
                logging.info(f"Attempting to load index: {selected_index_name} for chat.")
                # Pass current global Settings.llm and Settings.embed_model to load_index for cache consistency
                index = load_index(selected_index_name, Settings.llm, Settings.embed_model)
                if index:
                    st.session_state.chat_engine = get_chat_engine(index) # Uses current global Settings.llm
                    st.session_state.current_engine_index = selected_index_name
                    # If the index changed, clear previous messages
                    if st.session_state.current_engine_index != selected_index_name and st.session_state.current_engine_index is not None:
                         st.session_state.messages = []
                    logging.info(f"Chat engine ready for index: {selected_index_name} using {llm_display_name}")
                else:
                    st.session_state.chat_engine = None # Explicitly set to None on failure
                    st.session_state.current_engine_index = None
                    st.warning(f"Could not load the selected index '{selected_index_name}'. Please try another one or check logs.")
    elif st.session_state.chat_engine is not None:
        # If no index is selected, or LLM/Embed model is not configured, clear any existing engine
        logging.info("No index selected or LLM/Embed model not configured. Clearing chat engine and messages.")
        st.session_state.chat_engine = None
        st.session_state.current_engine_index = None
        st.session_state.messages = []


    # --- Display Chat Interface ---
    if selected_index_name and st.session_state.chat_engine:
        st.info(f"Chatting with index: **{selected_index_name}** using **{selected_model_key}**")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        for i, node in enumerate(message["sources"]):
                            filename = node.metadata.get('filename', 'N/A')
                            page_label = node.metadata.get('page_label', 'N/A')
                            score = node.score if hasattr(node, 'score') and node.score is not None else 'N/A'
                            score_text = f"{score:.2f}" if isinstance(score, float) else str(score)
                            text_snippet = node.text[:300].replace('\n', ' ') + "..."
                            st.markdown(f"**Source {i+1}:** `{filename}` (Score: {score_text} Page: {page_label})")
                            st.caption(text_snippet)
    else:
        if not selected_index_name and authentication_status: # Only show if logged in
            st.info("Select an index from the sidebar to begin chatting.")
        elif not Settings.llm and authentication_status:
            st.warning("Chat model (LLM) is not configured. Please check sidebar selection and API keys.")
        elif not Settings.embed_model and authentication_status:
            st.warning("Embedding model is not configured. Please check API keys.")


    # --- Chat Input and Response Handling ---
    if prompt := st.chat_input(f"Ask about documents in '{selected_index_name}'..." if selected_index_name else "Select an index first..."):
        if not st.session_state.chat_engine:
            st.warning("Chat engine not ready. Please select a valid index and ensure the model is configured.")
        elif not Settings.llm: # Should be caught by chat_engine check too
            st.error("Chat model (LLM) is not configured properly.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Thinking..."):
                        logging.info(f"Streaming chat with prompt: '{prompt}' using engine: {id(st.session_state.chat_engine)}")
                        response_stream = st.session_state.chat_engine.stream_chat(prompt)
                        
                        response_container = st.empty()
                        full_response = ""
                        for token in response_stream.response_gen:
                            full_response += token
                            response_container.markdown(full_response + "▌")
                        response_container.markdown(full_response) # Final response

                        source_nodes = response_stream.source_nodes if hasattr(response_stream, 'source_nodes') else []
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response,
                            "sources": source_nodes
                        })
                        # Show sources to user
                        st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during chat: {e}")
                    logging.error(f"Chat error: {e}", exc_info=True)
                    error_message = f"Sorry, I encountered an error: {e}"
                    st.session_state.messages.append({"role": "assistant", "content": error_message, "sources": []})
                    # st.rerun() # To display the error message immediately

elif authentication_status is False:
    st.error('Username/password is incorrect')
    # The login form is automatically re-rendered by authenticator.login()

elif authentication_status is None:
    st.warning('Please enter your username and password')
    # The login form is automatically re-rendered by authenticator.login()
