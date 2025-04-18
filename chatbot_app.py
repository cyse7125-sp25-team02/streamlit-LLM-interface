# -*- coding: utf-8 -*-
"""
Streamlit application for querying course feedback using RAG.

Loads configuration from a .env file.
Includes LLM-based extraction of metadata filters from the user query. # <-- Comment remains, but functionality removed
If no filters are extracted, prompts user for more specific query after a pause. # <-- Behavior changed
If filters are extracted but yield no results, pauses before generating final answer. # <-- Behavior changed
Connects to Pinecone to retrieve relevant feedback chunks based on user query embeddings
(generated via Vertex AI 'text-embedding-004') and extracted metadata filters. # <-- Filtering removed
Uses Google Gemini ('models/gemini-2.0-flash-001') to generate an answer.
Uses a Service Account JSON key file for Google Cloud authentication.
Features sidebar 'New Chat' button and refined UI text.
Includes intent check for greetings and similarity threshold for relevance.
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
from typing import List, Dict, Optional, Tuple, Any # Added Any

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
# FILTER_EXTRACTION_MODEL_NAME = "models/gemini-2.0-flash-001" # No longer needed for filtering
GEMINI_ANSWER_MODEL_NAME = "models/gemini-2.0-flash-001"

# RAG Parameters
try:
    TOP_K = int(os.getenv("TOP_K", "5"))
except ValueError:
    logging.warning(f"Invalid TOP_K value in .env file. Defaulting to 5.")
    TOP_K = 5
# ADDED: Similarity Threshold (Tune this value based on testing)
# Scores typically range from 0 to 1. Higher means more similar.
# A value between 0.7 and 0.8 might be a good starting point.
try:
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
except ValueError:
    logging.warning(f"Invalid SIMILARITY_THRESHOLD value in .env file. Defaulting to 0.75.")
    SIMILARITY_THRESHOLD = 0.75


# Constants for Vertex AI Embedding Model
VERTEX_MODEL_NAME = "text-embedding-004"
VERTEX_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"

# Metadata fields we want to filter on # <-- List remains, but not used for filtering
FILTERABLE_METADATA_KEYS = [
    "instructor_name",
    "course_code",
    "course_name",
    "semester_term",
    "semester_year",
    "credit_hours"
]

# ADDED: List of simple greetings/phrases to handle directly
SIMPLE_GREETINGS = {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "bye", "goodbye"}


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
        # Set credentials only if they haven't been set or have changed
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
            # Handle potential variations in the return type of list_indexes()
            if isinstance(index_list_result, list): existing_indexes_list = index_list_result
            elif hasattr(index_list_result, 'names'):
                 if callable(index_list_result.names): names_result = index_list_result.names()
                 else: names_result = index_list_result.names # Handle attribute access
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

# Removed extract_filters_with_llm function as it's no longer needed

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

# Modified query_pinecone to remove filter_dict parameter usage internally
def query_pinecone(index: pinecone.Index, index_name: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]: # Added Any type hint
    """Queries Pinecone index using only vector similarity."""
    if index is None: st.error("Pinecone index not initialized."); return []
    if not query_embedding: st.error("Cannot query Pinecone without a query embedding."); return []
    # Only include necessary parameters for vector search
    query_params = {"vector": query_embedding, "top_k": top_k, "include_metadata": True}
    log_filter_msg = "Performing vector search without metadata filters." # Updated log message
    # Removed filter logic
    try:
        logging.info(f"Querying Pinecone index '{index_name}' with top_k={top_k}. {log_filter_msg}")
        results = index.query(**query_params)
        # Ensure matches is always a list
        matches = results.get('matches', []) if results else []
        logging.info(f"Pinecone query returned {len(matches)} matches.")
        return matches
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}", exc_info=True)
        st.error(f"Failed to query Pinecone: {e}")
        return []

# MODIFIED format_context function (No changes needed here from previous version)
def format_context(matches: List[Dict[str, Any]]) -> str:
    """
    Formats the retrieved Pinecone matches into a context string for the LLM.
    Includes full metadata for each snippet as requested.
    """
    context = ""
    # Message updated slightly to reflect potential threshold filtering
    if not matches: return "No sufficiently relevant feedback found matching the query."

    context += "Relevant Course Feedback Snippets:\n"
    context += "------------------------------------\n"
    for i, match in enumerate(matches):
        metadata = match.get('metadata', {})
        score = match.get('score', 0.0) # Similarity score

        # Extract all relevant metadata fields for formatting
        course_name = metadata.get('course_name', 'N/A')
        course_code = metadata.get('course_code', 'N/A')
        instructor_name = metadata.get('instructor_name', 'N/A')
        semester_term = metadata.get('semester_term', 'N/A')
        semester_year = metadata.get('semester_year', 'N/A')
        credit_hours = metadata.get('credit_hours', 'N/A')
        question = metadata.get('question', 'N/A')
        # Use 'original_text' for the feedback content
        feedback_text = metadata.get('original_text', 'N/A')

        # Construct the detailed snippet string
        context += f"Snippet {i+1} (Score: {score:.4f}):\n"
        context += f"Course: {course_name} ({course_code})\n"
        context += f"Instructor: {instructor_name}\n"
        context += f"Semester: {semester_term} {semester_year}\n"
        context += f"Credit Hours: {credit_hours}\n"
        context += f"Question Context: {question}\n" # Changed label slightly for clarity
        context += f"Feedback Text: {feedback_text}\n" # Displays original text with newlines
        context += "------------------------------------\n"
    return context

def generate_answer_with_gemini(query: str, context: str, gemini_configured: bool) -> Optional[str]:
    """Generates an answer using Gemini based on the query and context."""
    if not gemini_configured: st.error("Gemini not configured. Cannot generate answer."); return None
    # The prompt instructions remain the same, asking the LLM to synthesize from the provided snippets
    prompt = f"""
You are a helpful assistant analyzing course feedback. Answer the following user query based *only* on the provided relevant course feedback snippets.

Instructions:
Do not use any prior knowledge or information outside of these snippets.
Do not refer/cite specific snippet numbers (e.g., "Snippet 1") in your answer.
Do not use words "snippet" or "snippets" in your answer, rather you can use "feedback" or "comments".
Do not just list individual comments one by one (e.g., avoid saying "one person/feedback/student said X, another person/feedback/student said Y"), rather summarize the snippets.
If the snippets do not contain enough information to answer the query, explicitly state that. If the query is a simple greeting or unrelated chit-chat, respond appropriately without mentioning the snippets.

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

        # Handle response variations
        if hasattr(response, 'text'): return response.text
        elif response.parts: return "".join(part.text for part in response.parts)
        # Check for blocked response or empty candidates
        elif not response.candidates:
             logging.warning("Gemini response blocked or empty.")
             # Check safety ratings if available (example structure, adjust based on actual API)
             try:
                 if response.prompt_feedback.block_reason:
                      logging.warning(f"Response blocked due to: {response.prompt_feedback.block_reason}")
                      return f"My response was blocked due to safety settings ({response.prompt_feedback.block_reason}). I cannot provide an answer based on the context."
             except AttributeError:
                 pass # No block reason info available
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
st.caption("Ask questions about course feedback. Mention correct information on courses, instructors, etc., to get proper results.") # Caption remains relevant

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
        # Ensure all necessary config vars are present for Vertex AI init
        if SERVICE_ACCOUNT_KEY_PATH and GCP_PROJECT_ID and GCP_LOCATION:
            st.session_state.vertex_model = get_vertex_embedding_model(GCP_PROJECT_ID, GCP_LOCATION, SERVICE_ACCOUNT_KEY_PATH)
            if not st.session_state.vertex_model: init_errors = True
        else: st.sidebar.error("GCP config (Project ID, Location, Service Account Path) missing in .env"); init_errors = True
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
            # Removed filter display logic
            # Display context only if it's not the "No relevant feedback" message
            # The context now contains the full metadata per snippet
            # Added check for "context_used" flag to avoid showing context for greetings
            if message.get("context_used", False) and message.get("context") and not message.get("context", "").startswith("No relevant"):
                 with st.expander("Retrieved Context (Full Details)"): # Updated expander title
                      st.text(message["context"]) # Display the fully formatted context

# Get user input
user_query = st.chat_input("Ask about course feedback... (e.g., How is Network Structures and Cloud Computing course?)")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"): st.markdown(user_query)

    # Check if services are initialized
    vertex_ok = st.session_state.get("vertex_model_initialized", False)
    pinecone_ok = st.session_state.get("pinecone_initialized", False)
    gemini_ok = st.session_state.get("gemini_configured", False)
    vertex_model_obj = st.session_state.get("vertex_model")
    pinecone_index_obj = st.session_state.get("pinecone_index")

    # --- MODIFIED: Intent Check and RAG Pipeline ---
    if vertex_ok and pinecone_ok and gemini_ok and vertex_model_obj and pinecone_index_obj:

        # 1. Intent Check for simple greetings
        normalized_query = user_query.lower().strip().rstrip('?.!')
        if normalized_query in SIMPLE_GREETINGS:
            logging.info(f"Detected simple greeting: '{user_query}'. Skipping RAG.")
            # Provide a direct response without hitting Pinecone or Gemini RAG
            if normalized_query in {"thanks", "thank you"}:
                assistant_response_content = "You're welcome! Let me know if you have questions about course feedback."
            elif normalized_query in {"bye", "goodbye"}:
                assistant_response_content = "Goodbye!"
            else: # Default greeting response
                assistant_response_content = "Hello! How can I help you with course feedback today?"
            retrieved_context_str = "No context needed for greeting."
            context_was_used = False # Flag to prevent showing context expander

        else: # Not a simple greeting, proceed with RAG pipeline
            with st.spinner("Thinking..."):
                assistant_response_content = None
                retrieved_context_str = f"No feedback found with similarity above {SIMILARITY_THRESHOLD}." # Default context
                matches = [] # Initialize matches
                context_was_used = False # Default to false

                # Step 1: Generate query embedding
                logging.info(f"Step 1/4: Generating query embedding...")
                query_embedding = get_query_embedding(user_query, vertex_model_obj)

                if query_embedding:
                    # Step 2: Query vector database
                    logging.info(f"Step 2/4: Querying vector database...")
                    matches = query_pinecone(
                        index=pinecone_index_obj,
                        index_name=PINECONE_INDEX_NAME,
                        query_embedding=query_embedding,
                        top_k=TOP_K
                    )

                    # Step 3: Check Similarity Threshold and Format Context
                    if matches:
                        top_score = matches[0].get('score', 0.0)
                        logging.info(f"Top match score: {top_score:.4f} (Threshold: {SIMILARITY_THRESHOLD})")
                        if top_score >= SIMILARITY_THRESHOLD:
                            logging.info(f"Step 3/4: Formatting context (Score >= Threshold)...")
                            retrieved_context_str = format_context(matches)
                            context_was_used = True # Mark that relevant context was found and formatted
                        else:
                             logging.info(f"Top match score below threshold. Discarding context.")
                             # Keep retrieved_context_str as the default "No feedback found..." message
                             matches = [] # Treat as no matches found for context formatting/display
                             context_was_used = False
                    else:
                        # No matches found from Pinecone
                        logging.info("Vector search returned no matches.")
                        retrieved_context_str = "No relevant feedback found matching the query."
                        context_was_used = False


                    # Step 4: Generate answer (always called, but context varies)
                    logging.info("Step 4/4: Generating answer...")
                    # Pass the potentially modified context string to Gemini
                    assistant_response_content = generate_answer_with_gemini(
                        user_query, retrieved_context_str, gemini_ok
                    )

                else: # Embedding generation failed
                    st.error("Could not generate embedding for the query.")
                    assistant_response_content = "Sorry, I encountered an error processing your query (embedding failed)."
                    retrieved_context_str = "Search skipped due to embedding error."
                    context_was_used = False

        # Add assistant message to history
        assistant_message = {
            "role": "assistant",
            "content": assistant_response_content if assistant_response_content else "Sorry, I encountered an error generating the response.",
            "context": retrieved_context_str, # Store context (or relevant message)
            "context_used": context_was_used # Store flag for UI display logic
        }
        st.session_state.messages.append(assistant_message)
        st.rerun() # Rerun to display the new message

    else: # Services not initialized
        st.warning("Services not initialized. Please check your `.env` file and restart the application.")
        # Clean up user message if services fail after input
        if st.session_state.messages:
             if st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
