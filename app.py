# app.py
import streamlit as st
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv
import data_processor_10th

load_dotenv()
st.set_page_config(page_title="MVA Results Dashboard", layout="wide")

def check_password():
    expected_password = os.getenv("ADMIN_PASSWORD") or expected_password == "admin123"
    if not expected_password:
        st.error("Admin password not configured.")
        return False

    def password_entered():
        if st.session_state.get("password_input") == expected_password:
            st.session_state["password_correct"] = True
            st.session_state["auth_message_shown"] = True
        else:
            st.session_state["password_correct"] = False
        if "password_input" in st.session_state:
            del st.session_state["password_input"]

    if "password_correct" not in st.session_state:
        with st.form("password_form"):
            st.text_input("Admin Password", type="password", key="password_input")
            submitted = st.form_submit_button("Submit")
        if submitted:
            password_entered()
            return st.session_state.get("password_correct", False)
        st.info("Please enter the admin password.")
        return False

    elif not st.session_state["password_correct"]:
        with st.form("password_form_retry"):
            st.text_input("Admin Password", type="password", key="password_input_retry")
            submitted = st.form_submit_button("Submit")
        if submitted and st.session_state.get("password_input_retry") == expected_password:
            st.session_state["password_correct"] = True
            st.session_state["auth_message_shown"] = True
            return True
        st.error("😕 Password incorrect")
        return False

    else:
        if not st.session_state.get("auth_message_shown", False):
            st.success("✅ Authentication Successful!")
            st.session_state["auth_message_shown"] = True
        return True

# --- Main App ---
try:
    if check_password():
        db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME', 'mva_analytics')
        }

        if 'loaded_analysis' in st.session_state:
            data = st.session_state['loaded_analysis']
            data_processor_10th.render_full_dashboard(data['df'], data['results'], data.get('models', {}))
            st.stop()

        st.subheader("📁 Load Previous Analysis (Optional)")
        past_runs = data_processor_10th.fetch_past_runs_from_db(db_config)
        if not past_runs.empty:
            run_options = dict(zip(past_runs['run_id'], past_runs['timestamp']))
            selected_run = st.selectbox("Select a previous analysis run:", options=list(run_options.keys()),
                                        format_func=lambda x: f"{run_options[x]} (ID: {x})")
            if st.button("Load Selected Analysis"):
                df_loaded, results_loaded, models_loaded = data_processor_10th.load_from_sql_by_run_id(selected_run, db_config)
                if df_loaded is not None:
                    st.session_state['loaded_analysis'] = {'df': df_loaded, 'results': results_loaded, 'models': models_loaded or {}}
                    st.rerun()
                else:
                    st.error("Failed to load selected analysis.")

        st.subheader("⬆️ Upload New Data for Analysis")
        class_group = st.selectbox("Select Class Group:", options=["8th to 10th", "11th", "12th"], key="class_group_selector")

        if class_group == "8th to 10th":
            uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx", key="excel_uploader")
            if uploaded_file:
                with st.spinner("Processing uploaded file..."):
                    temp_file_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_file_path = tmp_file.name

                        df_parsed, _ = data_processor_10th.load_and_parse_10th_sheet(temp_file_path)
                        if df_parsed.empty:
                            st.error("❌ No student data parsed.")
                        else:
                            new_data_hash = data_processor_10th.generate_academic_data_hash(df_parsed)
                            existing_run = data_processor_10th.check_data_exists_in_db(new_data_hash, db_config)
                            if existing_run:
                                st.warning(f"⚠️ This dataset already exists (Run ID: {existing_run}).")
                                overwrite = st.checkbox("Overwrite existing analysis in database?", key="overwrite_db")
                                if not overwrite:
                                    st.info("Loading existing analysis from database...")
                                    df_loaded, results_loaded, models_loaded = data_processor_10th.load_from_sql_by_run_id(existing_run, db_config)
                                    if df_loaded is not None:
                                        data_processor_10th.render_full_dashboard(df_loaded, results_loaded, models_loaded)
                                        st.stop()
                                    else:
                                        st.error("Failed to load existing data.")

                            analysis_results, trained_models = data_processor_10th.perform_advanced_analysis(df_parsed)
                            data_processor_10th.process_and_save_to_sql(df_parsed, analysis_results, trained_models, db_config, new_data_hash)
                            # ✅ Fixed: Use df_parsed, not df_loaded
                            data_processor_10th.render_full_dashboard(df_parsed, analysis_results, trained_models)
                            st.stop()
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                    finally:
                        if temp_file_path and os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
            else:
                st.info("Upload an Excel file or load a previous analysis above.")
        else:
            st.write("Database integration for 11th/12th is planned for future releases.")
    else:
        pass

except Exception as e:
    st.error(f"🚨 Unexpected error: {e}")
    import traceback
    st.code(traceback.format_exc())