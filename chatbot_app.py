# -*- coding: utf-8 -*-
"""
Streamlit application for querying course feedback using RAG.

Loads configuration from a .env file.
Includes LLM-based extraction of metadata filters from the user query.
If no filters are extracted, prompts user for more specific query after a pause.
If filters are extracted but yield no results, pauses before generating final answer.
Connects to Pinecone to retrieve relevant feedback chunks based on user query embeddings
(generated via Vertex AI 'text-embedding-004') and extracted metadata filters.
Uses Google Gemini ('models/gemini-2.0-flash-001') to generate an answer.
Uses a Service Account JSON key file for Google Cloud authentication.
Features sidebar 'New Chat' button and refined UI text.
"""

import os
import re
import pinecone
import google.generativeai as genai
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
import logging
import json # For parsing LLM response
import streamlit as st # Import Streamlit
from dotenv import load_dotenv # Import dotenv
import time # Ensure time is imported
from typing import List, Dict, Optional, Tuple

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file into environment

# --- Configuration & Constants (Loaded from Environment) ---

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GCP Configuration
SERVICE_ACCOUNT_KEY_PATH = os.getenv("SERVICE_ACCOUNT_KEY_PATH")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FILTER_EXTRACTION_MODEL_NAME = "models/gemini-2.0-flash-001"
GEMINI_ANSWER_MODEL_NAME = "models/gemini-2.0-flash-001"

# RAG Parameters
try:
    TOP_K = int(os.getenv("TOP_K", "5"))
except ValueError:
    logging.warning(f"Invalid TOP_K value in .env file. Defaulting to 5.")
    TOP_K = 5
# NOTE: SIMILARITY_THRESHOLD constant and logic removed as requested previously.

# Constants for Vertex AI Embedding Model
VERTEX_MODEL_NAME = "text-embedding-004"
VERTEX_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# Metadata fields we want to filter on
FILTERABLE_METADATA_KEYS = [
    "instructor_name",
    "course_code",
    "course_name",
    "semester_term",
    "semester_year",
    "credit_hours"
]

# --- Helper Functions (Adapted for Streamlit) ---

@st.cache_resource(show_spinner="Initializing Vertex AI Embedding Model...")
def get_vertex_embedding_model(project_id: str, location: str, service_account_path: str) -> Optional[TextEmbeddingModel]:
    """Initializes and returns the Vertex AI TextEmbeddingModel using Service Account."""
    st.session_state.vertex_model_initialized = False
    if not project_id: st.error("GCP_PROJECT_ID not found in .env."); return None
    if not location: st.error("GCP_LOCATION not found in .env."); return None
    if not service_account_path: st.error("SERVICE_ACCOUNT_KEY_PATH not found in .env."); return None
    if not os.path.exists(service_account_path): st.error(f"Service Account Key file not found: {service_account_path}"); return None
    try:
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") != service_account_path:
             os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
             logging.info(f"Set GOOGLE_APPLICATION_CREDENTIALS to: {service_account_path}")
        logging.info(f"Initializing Vertex AI for project {project_id} in {location}...")
        aiplatform.init(project=project_id, location=location)
        logging.info(f"Loading Vertex AI TextEmbeddingModel: {VERTEX_MODEL_NAME}")
        model = TextEmbeddingModel.from_pretrained(VERTEX_MODEL_NAME)
        logging.info("Successfully loaded Vertex AI embedding model.")
        st.session_state.vertex_model_initialized = True
        return model
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI or load model: {e}", exc_info=True)
        st.error(f"Failed to initialize Vertex AI Embedding Model: {e}")
        return None

@st.cache_resource(show_spinner="Initializing Pinecone Connection...")
def init_pinecone(api_key: str, index_name: str) -> Optional[pinecone.Index]:
    """Initializes connection to Pinecone and returns the index object."""
    st.session_state.pinecone_initialized = False
    if not api_key: st.error("PINECONE_API_KEY not found in .env."); return None
    if not index_name: st.error("PINECONE_INDEX_NAME not found in .env."); return None
    try:
        logging.info(f"Initializing Pinecone client...")
        pc = pinecone.Pinecone(api_key=api_key)
        existing_indexes_list = None
        try:
            index_list_result = pc.list_indexes()
            if isinstance(index_list_result, list): existing_indexes_list = index_list_result
            elif hasattr(index_list_result, 'names'):
                 if callable(index_list_result.names): names_result = index_list_result.names()
                 else: names_result = index_list_result.names
                 if isinstance(names_result, list): existing_indexes_list = names_result
                 else: logging.error(f".names method/attribute did not return a list, type: {type(names_result)}"); return None
            else: logging.error(f"Unexpected type returned by pc.list_indexes(): {type(index_list_result)}."); return None
        except Exception as list_err: logging.error(f"Error calling pc.list_indexes() or accessing names: {list_err}", exc_info=True); return None

        if existing_indexes_list is None: logging.error("Failed to extract index list after checks."); return None
        logging.info(f"Existing Pinecone indexes found: {existing_indexes_list}")

        if index_name not in existing_indexes_list:
            logging.error(f"Pinecone index '{index_name}' does not exist in the list: {existing_indexes_list}")
            st.error(f"Pinecone index '{index_name}' not found. Available: {existing_indexes_list}")
            return None
        else:
            logging.info(f"Connecting to existing Pinecone index: {index_name}")
            index = pc.Index(index_name)
            st.session_state.pinecone_initialized = True
            return index
    except Exception as e:
        logging.error(f"Error during Pinecone initialization or connection: {e}", exc_info=True)
        st.error(f"Failed to connect to Pinecone: {e}")
        return None

def configure_gemini(api_key: str) -> bool:
    """Configures the Google Generative AI client."""
    st.session_state.gemini_configured = False
    if not api_key: st.error("GEMINI_API_KEY not found in .env."); return False
    try:
        genai.configure(api_key=api_key)
        logging.info("Google Generative AI configured.")
        st.session_state.gemini_configured = True
        return True
    except Exception as e:
        logging.error(f"Failed to configure Google Generative AI: {e}", exc_info=True)
        st.error(f"Failed to configure Google Gemini: {e}")
        return False

# --- Filter Extraction, Embedding, Querying, Formatting, Generation Functions ---

def extract_filters_with_llm(query: str, filter_keys: List[str], gemini_configured: bool) -> Optional[Dict]:
    """Uses LLM to extract potential filter values from a user query."""
    if not gemini_configured:
        logging.error("Gemini is not configured for filter extraction.")
        st.warning("Gemini not configured, cannot extract filters.")
        return None
    # Only attempt extraction if query is not just a greeting etc.
    if len(query.split()) < 3 and not any(key in query.lower() for key in ['instructor', 'course', 'semester', 'assignment', 'feedback']):
         logging.info(f"Query '{query}' seems too short or generic for filter extraction. Skipping.")
         return None

    prompt = f"""
Analyze the following user query about course feedback. Intelligently extract keywords for the following categories using your NLP knowledge. Respond ONLY with a valid JSON object containing the extracted keys and their corresponding values. If a value for a category is not mentioned, omit the key from the JSON object.

Ensure values for 'semester_year' and 'credit_hours' are numbers (integers). Other values should be strings.

Categories to extract:
- instructor_name: (string) The name of the instructor
- course_code: (string) The course code, typically letters followed by numbers (e.g., "CSYE 6225", "INFO 7390").
- course_name: (string) The name of the course
- semester_term: (string) The semester term (e.g., "Spring", "Fall", "Summer").
- semester_year: (integer) The specific year as a four-digit number (e.g., 2024, 2023).
- credit_hours: (integer) The number of credit hours as a number (e.g., 4, 3).

User Query: "{query}"

JSON Output:
"""
    try:
        logging.info(f"Attempting to extract filters from query: '{query}' using {FILTER_EXTRACTION_MODEL_NAME}")
        model = genai.GenerativeModel(FILTER_EXTRACTION_MODEL_NAME)
        response = model.generate_content(prompt)
        raw_response_text = None
        if hasattr(response, 'text'): raw_response_text = response.text
        elif response.parts: raw_response_text = "".join(part.text for part in response.parts)

        if not raw_response_text: logging.warning("LLM filter extraction returned empty response."); return None
        logging.info(f"Raw LLM filter extraction response: {raw_response_text}")

        json_match = re.search(r'\{.*\}', raw_response_text, re.DOTALL)
        if not json_match: logging.warning("Could not find JSON object in LLM response for filters."); return None

        json_string = json_match.group(0)
        extracted_data = json.loads(json_string)
        logging.info(f"Successfully parsed extracted filters: {extracted_data}")

        pinecone_filter = {}
        string_keys = ["instructor_name", "course_code", "course_name", "semester_term"]
        integer_keys = ["semester_year", "credit_hours"]

        for key, value in extracted_data.items():
            if key in filter_keys and value is not None and value != "":
                if key in string_keys:
                    try: pinecone_filter[key] = str(value).lower()
                    except Exception: logging.warning(f"Could not convert value for key '{key}' to string: {value}")
                elif key in integer_keys:
                    try: pinecone_filter[key] = int(value)
                    except (ValueError, TypeError): logging.warning(f"Could not convert value for key '{key}' to integer: {value}. Skipping.")

        if not pinecone_filter: logging.info("No relevant filters extracted or processed."); return None
        logging.info(f"Constructed Pinecone filter: {pinecone_filter}")
        return pinecone_filter
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON from LLM filter response: {raw_response_text}", exc_info=True)
        st.error("Could not parse filter information from LLM response.")
        return None
    except Exception as e:
        if "response was blocked" in str(e).lower():
             logging.warning(f"LLM filter extraction response blocked. Query: {query}")
             st.warning("Filter extraction was blocked. Proceeding without filters.")
             return None
        logging.error(f"Error during LLM filter extraction: {e}", exc_info=True)
        st.error(f"Failed to extract filters using LLM: {e}")
        return None

def get_query_embedding(query: str, model: TextEmbeddingModel) -> Optional[List[float]]:
    """Generates embedding for the user query using Vertex AI."""
    if model is None: st.error("Vertex AI embedding model not available."); return None
    try:
        logging.info(f"Generating embedding for query: '{query}'")
        input_data = [TextEmbeddingInput(text=query, task_type=VERTEX_TASK_TYPE_QUERY)]
        embeddings = model.get_embeddings(input_data)
        logging.info("Embedding generated successfully.")
        return embeddings[0].values
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}", exc_info=True)
        st.error(f"Failed to generate embedding for the query: {e}")
        return None

def query_pinecone(index: pinecone.Index, index_name: str, query_embedding: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Dict]:
    """Queries Pinecone index, optionally applying metadata filters."""
    if index is None: st.error("Pinecone index not initialized."); return []
    if not query_embedding: st.error("Cannot query Pinecone without a query embedding."); return []
    query_params = {"vector": query_embedding, "top_k": top_k, "include_metadata": True}
    log_filter_msg = "No metadata filter applied."
    if filter_dict:
        log_filter_msg = f"Applying metadata filter: {filter_dict}"
        query_params["filter"] = filter_dict
    try:
        logging.info(f"Querying Pinecone index '{index_name}' with top_k={top_k}. {log_filter_msg}")
        results = index.query(**query_params)
        matches = results.get('matches', []) if results else []
        logging.info(f"Pinecone query returned {len(matches)} matches.")
        return matches
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}", exc_info=True)
        st.error(f"Failed to query Pinecone: {e}")
        return []

def format_context(matches: List[Dict]) -> str:
    """Formats the retrieved Pinecone matches into a context string for the LLM."""
    context = ""
    if not matches: return "No relevant feedback found matching the criteria."
    context += "Relevant Course Feedback Snippets:\n"
    context += "------------------------------------\n"
    for i, match in enumerate(matches):
        metadata = match.get('metadata', {})
        text = metadata.get('original_text', 'N/A')
        question = metadata.get('question', 'N/A')
        score = match.get('score', 0.0)
        context += f"Snippet {i+1} (Score: {score:.4f}):\n"
        context += f"  Question Context: {question}\n"
        context += f"  Feedback Text: {text}\n"
        context += "------------------------------------\n"
    return context

def generate_answer_with_gemini(query: str, context: str, gemini_configured: bool) -> Optional[str]:
    """Generates an answer using Gemini based on the query and context."""
    if not gemini_configured: st.error("Gemini not configured. Cannot generate answer."); return None
    prompt = f"""
You are a helpful assistant analyzing course feedback. Answer the following user query based *only* on the provided relevant course feedback snippets.

Instructions:
Do not use any prior knowledge or information outside of these snippets.
Do not refer/cite specific snippet numbers (e.g., "Snippet 1") in your answer.
Do not use words "snippet" or "snippets" in your answer, rather you can use "feedback" or "comments".
Do not just list individual comments one by one (e.g., avoid saying "one person/feedback/student said X, another person/feedback/student said Y"), rather summarize the snippets.
If the snippets do not contain enough information to answer the query, explicitly state that.

User Query:
"{query}"

Relevant Course Feedback Snippets:
--- START CONTEXT ---
{context}
--- END CONTEXT ---

Answer:
"""
    try:
        logging.info(f"Generating answer with Gemini model: {GEMINI_ANSWER_MODEL_NAME}...")
        model = genai.GenerativeModel(GEMINI_ANSWER_MODEL_NAME)
        response = model.generate_content(prompt)
        logging.info("Gemini response received.")

        if hasattr(response, 'text'): return response.text
        elif response.parts: return "".join(part.text for part in response.parts)
        elif not response.candidates:
             logging.warning("Gemini response blocked or empty.")
             return "I cannot provide an answer based on the provided context, possibly due to safety settings or lack of information."
        else:
            logging.warning(f"Unexpected Gemini response structure: {response}")
            st.error("Could not parse Gemini response.")
            return None
    except Exception as e:
        logging.error(f"Error generating answer with Gemini: {e}", exc_info=True)
        if "API key not valid" in str(e): st.error("Invalid Gemini API Key.")
        elif "models/" in str(e) and "is not found" in str(e): st.error(f"Gemini model '{GEMINI_ANSWER_MODEL_NAME}' not found or not supported.")
        else: st.error(f"Failed to generate answer using Gemini: {e}")
        return None

# --- Streamlit App UI ---

st.set_page_config(page_title="Course Feedback RAG", layout="wide")
st.title("📚 Course Feedback Analysis (RAG + Gemini)")
# --- Updated Caption ---
st.caption("Ask questions about course feedback. Mention correct information on courses, instructors, etc., to get proper results.")

# --- Initialize Session State Variables ---
if "pinecone_index" not in st.session_state: st.session_state.pinecone_index = None
if "vertex_model" not in st.session_state: st.session_state.vertex_model = None
if "gemini_configured" not in st.session_state: st.session_state.gemini_configured = False
if "pinecone_initialized" not in st.session_state: st.session_state.pinecone_initialized = False
if "vertex_model_initialized" not in st.session_state: st.session_state.vertex_model_initialized = False
if "messages" not in st.session_state: st.session_state.messages = []
if "config_loaded" not in st.session_state: st.session_state.config_loaded = False
if "initialization_attempted" not in st.session_state: st.session_state.initialization_attempted = False

# --- Automatic Initialization on First Run / Rerun after clearing cache ---
if not st.session_state.initialization_attempted:
    st.session_state.initialization_attempted = True
    init_errors = False
    with st.spinner("Initializing connections from .env configuration..."):
        st.cache_resource.clear()
        if SERVICE_ACCOUNT_KEY_PATH and GCP_PROJECT_ID and GCP_LOCATION:
            st.session_state.vertex_model = get_vertex_embedding_model(GCP_PROJECT_ID, GCP_LOCATION, SERVICE_ACCOUNT_KEY_PATH)
            if not st.session_state.vertex_model: init_errors = True
        else: st.sidebar.error("GCP config missing in .env"); init_errors = True
        if PINECONE_API_KEY and PINECONE_INDEX_NAME:
            st.session_state.pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX_NAME)
            if not st.session_state.pinecone_index: init_errors = True
        else: st.sidebar.error("Pinecone config missing in .env"); init_errors = True
        if GEMINI_API_KEY:
            if not configure_gemini(GEMINI_API_KEY): init_errors = True
        else: st.sidebar.error("Gemini API Key missing in .env"); init_errors = True
    if init_errors:
         st.error("Initialization failed. Check .env file and logs, then refresh.")

# --- Sidebar ---
with st.sidebar:
    st.header("Chat Controls")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.success("New chat started!")
        time.sleep(0.5) # Keep brief pause
        st.rerun()
    st.divider()
    # Similarity threshold display removed

# --- Chat Interface ---

# Display previous messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("filters"):
                with st.expander("Applied Filters"): st.json(message["filters"])
            if message.get("context") and not message.get("context", "").startswith("No relevant") and not message.get("context", "").startswith("Search skipped"):
                 with st.expander("Retrieved Context"): st.text(message["context"])

# Get user input
user_query = st.chat_input("Ask about course feedback... (e.g., How is Network Structures and Cloud Computing course?)")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)

    vertex_ok = st.session_state.get("vertex_model_initialized", False)
    pinecone_ok = st.session_state.get("pinecone_initialized", False)
    gemini_ok = st.session_state.get("gemini_configured", False)
    vertex_model_obj = st.session_state.get("vertex_model")
    pinecone_index_obj = st.session_state.get("pinecone_index")

    if vertex_ok and pinecone_ok and gemini_ok and vertex_model_obj and pinecone_index_obj:
        with st.spinner("Thinking..."):
            assistant_response_content = None
            extracted_filter_dict = None
            retrieved_context_str = "No relevant feedback found." # Default context
            matches = [] # Initialize matches

            logging.info("Step 1/X: Extracting filters...")
            extracted_filter_dict = extract_filters_with_llm(
                user_query, FILTERABLE_METADATA_KEYS, gemini_ok
            )

            # --- Conditional Logic: Check if filters were extracted ---
            if extracted_filter_dict is None:
                logging.info("No filters extracted, providing default message.")
                # --- Add pause before showing default message ---
                time.sleep(2)
                # --- Update default message ---
                assistant_response_content = "Please mention some information on courses or instructors to get relevant results."
                retrieved_context_str = "Search skipped as no filters were extracted."
                # Skip embedding, query, format, generate steps
            else:
                # --- Proceed with RAG pipeline if filters ARE extracted ---
                log_filter_msg = f"Filters: {json.dumps(extracted_filter_dict)}"
                logging.info(f"Step 2/5: Generating query embedding... ({log_filter_msg})")
                query_embedding = get_query_embedding(user_query, vertex_model_obj)

                if query_embedding:
                    logging.info(f"Step 3/5: Querying vector database... ({log_filter_msg})")
                    matches = query_pinecone(
                        index=pinecone_index_obj,
                        index_name=PINECONE_INDEX_NAME,
                        query_embedding=query_embedding,
                        top_k=TOP_K,
                        filter_dict=extracted_filter_dict
                    )

                    # --- Add pause if query returned no matches ---
                    if not matches:
                        logging.info("Query with filters returned no matches. Pausing.")
                        time.sleep(2)
                    # --- End pause logic ---

                    # NOTE: No similarity score filtering applied here
                    final_matches_for_context = matches
                    logging.info(f"Retrieved {len(final_matches_for_context)} matches from Pinecone.")

                    logging.info(f"Step 4/5: Formatting context...")
                    retrieved_context_str = format_context(final_matches_for_context) # Will return "No relevant..." if matches is empty

                    logging.info("Step 5/5: Generating answer...")
                    assistant_response_content = generate_answer_with_gemini(
                        user_query, retrieved_context_str, gemini_ok
                    )
                else:
                    st.error("Could not generate embedding for the query.")
                    assistant_response_content = "Sorry, I encountered an error processing your query (embedding failed)."
            # --- End Conditional Logic ---

        # Add assistant message to history (either default or generated)
        assistant_message = {
            "role": "assistant",
            "content": assistant_response_content if assistant_response_content else "Sorry, I encountered an error generating the response.",
            "filters": extracted_filter_dict, # Store filters (even if None)
            "context": retrieved_context_str # Store context (or skip message)
        }
        st.session_state.messages.append(assistant_message)
        st.rerun()

    else:
        st.warning("Services not initialized. Please check your `.env` file and restart the application.")
        if st.session_state.messages:
             if st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
