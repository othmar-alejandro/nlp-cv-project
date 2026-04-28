"""HomeEstimator AI - Streamlit Application

A multimodal home service cost estimation system that combines
Computer Vision, Natural Language Processing, Speech-to-Text (Whisper),
and GPT-powered chat to analyze job photos and descriptions,
providing cost estimates and recommendations.
"""
import os
import sys
import streamlit as st
import pandas as pd
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.cv_pipeline import load_model as load_cv_model, predict_image
from src.nlp_pipeline import load_models as load_nlp_models, predict_text, extract_entities
from src.estimator import load_pricing_table, fuse_predictions, generate_estimate

# Page config
st.set_page_config(
    page_title="HomeEstimator AI",
    page_icon="🏠",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
.big-metric { font-size: 2.5rem; font-weight: bold; color: #2c3e50; }
.estimate-box { background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 2px solid #3498db; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load all models once and cache them."""
    cv_model = load_cv_model()
    cat_model, urg_model, tfidf = load_nlp_models()
    pricing = load_pricing_table()
    return cv_model, cat_model, urg_model, tfidf, pricing


def run_analysis(cv_model, cat_model, urg_model, tfidf, pricing, uploaded_image, job_description):
    """Run the full CV + NLP analysis pipeline."""
    cv_result = None
    if uploaded_image:
        uploaded_image.seek(0)
        cv_result = predict_image(cv_model, uploaded_image)

    nlp_result = None
    entities = {}
    if job_description:
        nlp_result = predict_text(cat_model, urg_model, tfidf, job_description)
        entities = extract_entities(job_description)

    # Handle single-input cases
    if cv_result and not nlp_result:
        nlp_result = {
            "category": cv_result["category"],
            "category_confidence": 0.5,
            "urgency": "medium",
            "urgency_confidence": 0.5,
        }
    elif nlp_result and not cv_result:
        cv_result = {
            "category": nlp_result["category"],
            "confidence": 0.5,
            "probabilities": {nlp_result["category"]: 0.5},
        }

    fused = fuse_predictions(cv_result, nlp_result, entities)
    estimate = generate_estimate(fused, pricing)

    return cv_result, nlp_result, fused, entities, estimate


def display_results(cv_result, nlp_result, fused, entities, estimate, job_description="", use_gpt=False):
    """Display the estimate results."""
    st.markdown("---")
    st.subheader("Estimate Results")

    # Top metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Category", estimate.get("category", "N/A").replace("_", " ").title())
    with col_b:
        st.metric("Urgency", estimate.get("urgency", "N/A").title())
    with col_c:
        st.metric("Scope", estimate.get("scope", "N/A").title())
    with col_d:
        conf = estimate.get("confidence", 0)
        st.metric("Confidence", f"{conf:.0%}")

    # Price estimate
    st.markdown("### Estimated Cost Range")
    price_low = estimate.get("price_low", "N/A")
    price_high = estimate.get("price_high", "N/A")
    if isinstance(price_low, (int, float)):
        st.markdown(f"## ${price_low:,} — ${price_high:,}")
    else:
        st.markdown("## Price estimate unavailable")

    # GPT-powered smart analysis
    if use_gpt and os.getenv("OPENAI_API_KEY"):
        try:
            from src.chat_pipeline import generate_smart_estimate
            with st.spinner("Generating AI-powered analysis..."):
                smart_response = generate_smart_estimate(
                    estimate, entities,
                    original_text=job_description,
                    image_category=cv_result.get("category", "") if cv_result else "",
                )
            st.markdown("### AI Analysis")
            st.markdown(smart_response)
        except Exception as e:
            st.warning(f"GPT analysis unavailable: {e}")

    # Rule-based details
    with st.expander("Typical Tasks for This Type of Job", expanded=not use_gpt):
        st.write(estimate.get("typical_tasks", "N/A"))

    with st.expander("Recommendations", expanded=not use_gpt):
        st.write(estimate.get("recommendations", "N/A"))

    with st.expander("Suggested Next Steps", expanded=not use_gpt):
        st.write(estimate.get("next_steps", "N/A"))

    # Extracted entities
    if entities and any(entities.values()):
        with st.expander("Extracted Details from Description"):
            if entities.get("measurements"):
                st.write(f"**Measurements:** {', '.join(entities['measurements'])}")
            if entities.get("materials"):
                st.write(f"**Materials:** {', '.join(entities['materials'])}")
            if entities.get("locations"):
                st.write(f"**Locations:** {', '.join(entities['locations'])}")
            if entities.get("quantities"):
                st.write(f"**Quantities:** {', '.join(entities['quantities'])}")

    # Model analysis details
    with st.expander("Model Analysis Details"):
        st.write(f"**CV Model Prediction:** {cv_result['category'].replace('_', ' ').title()} "
                 f"({cv_result['confidence']:.1%})")
        if "category_confidence" in nlp_result:
            st.write(f"**NLP Model Prediction:** {nlp_result['category'].replace('_', ' ').title()} "
                     f"({nlp_result['category_confidence']:.1%})")
        agreement = "Yes" if fused.get("agreement") else "No"
        st.write(f"**Models Agree:** {agreement}")

    return estimate


def main():
    # Header
    st.title("HomeEstimator AI")
    st.markdown("*Multimodal home service cost estimation powered by CV, NLP, Whisper & GPT*")
    st.markdown("---")

    # Load models
    try:
        cv_model, cat_model, urg_model, tfidf, pricing = load_all_models()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training notebooks first to generate the model files.")
        models_loaded = False

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_estimate" not in st.session_state:
        st.session_state.last_estimate = None

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Get Estimate", "Chat with AI", "How It Works", "Model Performance"])

    # ===== TAB 1: GET ESTIMATE =====
    with tab1:
        if not models_loaded:
            st.warning("Models not loaded. Please train models first.")
            return

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Upload Job Photo")
            uploaded_image = st.file_uploader(
                "Upload an image of the job/issue",
                type=["jpg", "jpeg", "png"],
                help="Take a photo of the area that needs work",
            )
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.subheader("Describe the Job")

            # Voice input
            st.markdown("**Option 1: Voice Input (Whisper)**")
            try:
                from st_audiorec import st_audiorec
                audio_bytes = st_audiorec()
                if audio_bytes:
                    with st.spinner("Transcribing with Whisper..."):
                        from src.voice_pipeline import transcribe_audio
                        result = transcribe_audio(audio_bytes)
                        transcribed_text = result["text"]
                    st.success(f"Transcribed: *{transcribed_text}*")
                else:
                    transcribed_text = ""
            except ImportError:
                st.info("Voice input requires streamlit-audiorec. Install with: pip install streamlit-audiorec")
                transcribed_text = ""

            st.markdown("**Option 2: Type your description**")
            typed_text = st.text_area(
                "What work needs to be done?",
                value=transcribed_text if transcribed_text else "",
                placeholder="Example: The kitchen faucet has been dripping for a week...",
                height=150,
            )
            job_description = typed_text

            # GPT toggle
            use_gpt = st.checkbox("Use GPT for smart analysis", value=True,
                                   help="Uses OpenAI GPT to generate detailed analysis and recommendations")

        # Analyze button
        st.markdown("---")
        analyze_button = st.button("Get Estimate", type="primary", use_container_width=True)

        if analyze_button:
            if not uploaded_image and not job_description:
                st.warning("Please upload an image and/or provide a job description.")
                return

            with st.spinner("Analyzing your request..."):
                cv_result, nlp_result, fused, entities, estimate = run_analysis(
                    cv_model, cat_model, urg_model, tfidf, pricing, uploaded_image, job_description
                )

            result = display_results(cv_result, nlp_result, fused, entities, estimate,
                                      job_description=job_description, use_gpt=use_gpt)

            # Save for chat context
            st.session_state.last_estimate = estimate
            st.session_state.chat_history = [
                {"role": "assistant", "content": f"I analyzed your request. "
                 f"Category: {estimate.get('category', 'N/A')}, "
                 f"Urgency: {estimate.get('urgency', 'N/A')}, "
                 f"Estimated cost: ${estimate.get('price_low', 'N/A')}-${estimate.get('price_high', 'N/A')}. "
                 f"Feel free to ask me any follow-up questions!"}
            ]

    # ===== TAB 2: CHAT WITH AI =====
    with tab2:
        st.subheader("Chat with HomeEstimator AI")

        if not os.getenv("OPENAI_API_KEY"):
            st.warning("OpenAI API key not found. Add it to .env file to enable chat.")
            return

        if st.session_state.last_estimate:
            st.info(f"Chatting about: **{st.session_state.last_estimate.get('category', 'N/A').replace('_', ' ').title()}** job "
                    f"(${st.session_state.last_estimate.get('price_low', '?')}-${st.session_state.last_estimate.get('price_high', '?')})")
        else:
            st.info("Get an estimate first in the 'Get Estimate' tab, then come here to ask follow-up questions. "
                    "Or just ask any home service question!")

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a follow-up question...")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from src.chat_pipeline import chat_followup
                        response = chat_followup(st.session_state.chat_history[:-1], user_input)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I couldn't process that: {e}"
                        st.error(error_msg)

    # ===== TAB 3: HOW IT WORKS =====
    with tab3:
        st.subheader("System Architecture")
        st.markdown("""
### How HomeEstimator AI Works

HomeEstimator AI is a **multimodal pipeline** that combines multiple AI technologies:

#### Pipeline Overview:
1. **Image Analysis (CV)**: MobileNetV2 with transfer learning classifies uploaded photos into 6 job categories
2. **Text Analysis (NLP)**: TF-IDF + Logistic Regression classifies text by category and urgency
3. **Voice Input (Whisper)**: OpenAI Whisper converts speech to text for hands-free input
4. **Entity Extraction**: spaCy + regex patterns extract measurements, materials, locations, and quantities
5. **Signal Fusion**: CV and NLP predictions combined using weighted voting
6. **Smart Analysis (GPT)**: OpenAI GPT generates detailed, natural language estimates and recommendations
7. **Chat**: Follow-up conversation for clarifying questions

#### Job Categories:
- **Plumbing**: Pipes, faucets, water heaters, drains, toilets
- **Painting**: Interior/exterior painting, staining, wall repair
- **Roofing**: Shingles, leaks, gutters, flashing, structural
- **Electrical**: Outlets, panels, wiring, lighting, circuits
- **HVAC**: Heating, cooling, ductwork, thermostats, ventilation
- **General Repair**: Drywall, doors, windows, fences, decks

#### Urgency Levels:
- **Low**: Routine maintenance
- **Medium**: Address within days
- **High**: Active damage or safety concern
- **Emergency**: Immediate response needed

#### Technologies Used:
| Component | Technology |
|-----------|-----------|
| Image Classification | PyTorch + MobileNetV2 (Transfer Learning) |
| Text Classification | scikit-learn (TF-IDF + Logistic Regression) |
| Entity Extraction | spaCy + Regex |
| Voice Input | OpenAI Whisper (base model) |
| Smart Analysis | OpenAI GPT-4o-mini |
| Web Application | Streamlit |
        """)

    # ===== TAB 4: MODEL PERFORMANCE =====
    with tab4:
        st.subheader("Model Performance Metrics")
        figure_dir = os.path.join(PROJECT_ROOT, "report", "figures")
        metrics_figures = [
            ("Text Category Classifier - Confusion Matrix", "text_category_confusion_matrix.png"),
            ("Text Urgency Classifier - Confusion Matrix", "text_urgency_confusion_matrix.png"),
            ("Top TF-IDF Features per Category", "text_feature_importance.png"),
            ("Image Classifier - Confusion Matrix", "image_confusion_matrix.png"),
            ("Image Classifier - Training Curves", "image_training_curves.png"),
        ]
        for title, filename in metrics_figures:
            fig_path = os.path.join(figure_dir, filename)
            if os.path.exists(fig_path):
                st.markdown(f"#### {title}")
                st.image(fig_path, use_container_width=True)
                st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### About")
        st.markdown("""
**HomeEstimator AI** combines Computer Vision,
NLP, Speech Recognition, and GPT to provide
quick cost estimates for home service jobs.

Upload a photo + describe the issue (type or speak)
to get an instant estimate with AI recommendations.
        """)

        st.markdown("### Tech Stack")
        st.markdown("""
- **CV**: MobileNetV2 (Transfer Learning)
- **NLP**: TF-IDF + Logistic Regression
- **Voice**: OpenAI Whisper
- **Chat**: OpenAI GPT-4o-mini
- **Entities**: spaCy + Regex
- **App**: Streamlit
        """)

        st.markdown("---")
        st.markdown("*Built for CV & NLP Final Project*")


if __name__ == "__main__":
    main()
