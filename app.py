import streamlit as st
import pandas as pd
import os
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
CSV_FILE = 'fellow_log.csv'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- Helper Functions ---
def load_log():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=['Date', 'Topic', 'Procedure', 'Key Learning Points'])
    return df

def save_log(df):
    df.to_csv(CSV_FILE, index=False)

def get_openai_recommendations(log_df):
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return "Error: OpenAI API key not set."

    client = OpenAI(api_key=OPENAI_API_KEY)

    log_text = log_df.to_string(index=False)
    prompt = f"""
    You are an expert in PCCM (Pulmonary, Critical Care, and Sleep Medicine) fellowship training.
    Analyze the following log of a fellow's experiences and provide recommendations for next steps to ensure comprehensive coverage of all essential PCCM topics and procedures.
    Focus on identifying gaps, suggesting areas for further experience, and proposing specific learning objectives.

    Fellowship Log:
    {log_text}

    Recommendations:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides recommendations for PCCM fellowship training."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with OpenAI: {e}")
        return f"Error: {e}"

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="PCCM Fellowship Log & Analyzer")

st.title("PCCM Fellowship Experience Log")

# Sidebar for new entry
st.sidebar.header("Record New Experience")
with st.sidebar.form("new_experience_form"):
    date = st.date_input("Date", datetime.now())
    topic = st.text_input("Topic (e.g., ARDS, Sepsis, Asthma)")
    procedure = st.text_input("Procedure (e.g., Bronchoscopy, Central Line, Intubation)")
    key_learning_points = st.text_area("Key Learning Points")

    submitted = st.form_submit_button("Record Experience")
    if submitted:
        if not all([date, topic, key_learning_points]):
            st.error("Please fill in Date, Topic, and Key Learning Points.")
        else:
            new_entry = pd.DataFrame([{
                'Date': date.strftime('%Y-%m-%d'),
                'Topic': topic,
                'Procedure': procedure,
                'Key Learning Points': key_learning_points
            }])
            current_log = load_log()
            updated_log = pd.concat([current_log, new_entry], ignore_index=True)
            save_log(updated_log)
            st.sidebar.success("Experience recorded successfully!")
            st.rerun() # Rerun to update the main display

# Main area for log display and analysis
st.header("Your Fellowship Log")
log_df = load_log()
if not log_df.empty:
    st.dataframe(log_df, use_container_width=True)
else:
    st.info("No experiences logged yet. Use the sidebar to record your first experience!")

st.header("Analyze Log with GPT-4o")
if st.button("Get Recommendations"):
    with st.spinner("Analyzing log and fetching recommendations..."):
        if not log_df.empty:
            recommendations = get_openai_recommendations(log_df)
            st.subheader("GPT-4o Recommendations:")
            st.write(recommendations)
        else:
            st.warning("Log is empty. Please record some experiences before analysis.")
