import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# ============================================
# CONFIGURATION — Replace with your Groq API Key
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# ============================================

st.set_page_config(page_title="TN Scheme Finder 2026", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.header-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 25px;
    border: 1px solid #e94560;
}
.header-box h1 { color: #ffffff; font-size: 2rem; margin: 0; }
.header-box p { color: #a8b2d8; margin: 8px 0 0 0; font-size: 1rem; }
.stat-box {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
.stat-box h3 { color: #e94560; font-size: 1.8rem; margin: 0; }
.stat-box p { color: #a8b2d8; margin: 5px 0 0 0; font-size: 0.85rem; }
.section-title {
    color: #e94560;
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 2px solid #e94560;
}
.result-box {
    background: #1e1e2e;
    border-left: 4px solid #e94560;
    border-radius: 8px;
    padding: 20px;
    margin-top: 10px;
    color: #e0e0e0;
}
.chat-user {
    background: #0f3460;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 8px 0;
    color: #ffffff;
}
.chat-ai {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 10px 15px;
    margin: 8px 0;
    border-left: 3px solid #e94560;
    color: #e0e0e0;
}
.disclaimer {
    background: #1a1a2e;
    border: 1px solid #555;
    border-radius: 8px;
    padding: 10px 15px;
    font-size: 0.8rem;
    color: #888;
    margin-top: 15px;
}
.info-box {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 20px;
    color: #a8b2d8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>🏛️ Tamil Nadu Government Scheme Finder 2026</h1>
    <p>Official AI-powered portal — Find all welfare schemes you are eligible for | தமிழ் மற்றும் English</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><h3>25+</h3><p>Real TN Schemes 2026</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><h3>10+</h3><p>Categories</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><h3>2</h3><p>Languages</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><h3>AI</h3><p>Powered Matching</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

@st.cache_resource
def load_schemes():
    loader = TextLoader("schemes_data.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "scheme_result" not in st.session_state:
    st.session_state.scheme_result = ""
if "user_profile" not in st.session_state:
    st.session_state.user_profile = ""

left_col, right_col = st.columns([1, 1.3])

with left_col:
    st.markdown('<div class="section-title">📋 உங்கள் விவரங்களை உள்ளிடவும் / Enter Your Details</div>', unsafe_allow_html=True)

    with st.form("user_details"):
        user_name = st.text_input("பெயர் / Full Name", placeholder="e.g. Vishwa Kumar")
        language = st.selectbox("மொழி / Response Language", ["English", "Tamil"])

        st.markdown("**👤 Personal Information**")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("வயது / Age", min_value=1, max_value=100, value=22)
            gender = st.selectbox("பாலினம் / Gender", ["Male", "Female", "Other"])
        with col2:
            category = st.selectbox("சமுதாயம் / Community", ["General", "BC", "MBC", "SC", "ST", "OBC"])
            marital_status = st.selectbox("திருமண நிலை / Marital Status", ["Unmarried", "Married", "Widow", "Widower", "Divorced"])

        st.markdown("**💰 Economic Information**")
        col3, col4 = st.columns(2)
        with col3:
            income = st.number_input("வருடாந்திர வருமானம் / Annual Income (Rs)", min_value=0, value=50000, step=5000)
            occupation = st.selectbox("தொழில் / Occupation", [
                "Student", "Unemployed Graduate", "Unemployed",
                "Agricultural Labourer", "Farmer", "Artisan",
                "Pregnant Woman", "Senior Citizen", "Differently Abled",
                "Government Employee", "Private Employee", "Other"
            ])
        with col4:
            district = st.selectbox("மாவட்டம் / District", [
                "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli",
                "Salem", "Tirunelveli", "Vellore", "Erode", "Thoothukudi",
                "Dindigul", "Thanjavur", "Ranipet", "Sivaganga",
                "Virudhunagar", "Nagapattinam", "Cuddalore", "Villupuram",
                "Kancheepuram", "Chengalpattu", "Tiruppur", "Nilgiris",
                "Dharmapuri", "Krishnagiri", "Namakkal", "Perambalur",
                "Ariyalur", "Karur", "Tiruvarur", "Mayiladuthurai",
                "Pudukkottai", "Ramanathapuram", "Tenkasi", "Theni",
                "Tirupathur", "Kallakurichi", "Kanniyakumari"
            ])
            residence = st.selectbox("வாழும் இடம் / Residence", ["Urban", "Rural", "Semi-Urban"])

        st.markdown("**📄 Additional Details**")
        col5, col6 = st.columns(2)
        with col5:
            ration_card = st.selectbox("ரேஷன் கார்டு / Ration Card", ["No card", "White", "Green (BPL)", "Yellow (AAY)"])
            education = st.selectbox("கல்வி / Education Level", [
                "No formal education", "Primary (up to 5th)",
                "Middle (up to 8th)", "High School (10th)",
                "Higher Secondary (12th)", "Diploma / ITI",
                "Graduate", "Post Graduate"
            ])
        with col6:
            is_govt_school = st.selectbox("Studied in Govt School?", ["Yes", "No", "Not Applicable"])
            has_disability = st.selectbox("Disability Status", ["No disability", "40% or more disability", "Below 40% disability"])

        submitted = st.form_submit_button("🔍 Find My Eligible Schemes / திட்டங்கள் கண்டறிய", use_container_width=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> This tool uses AI to match schemes. Always verify eligibility at official 
    Tamil Nadu Government websites before applying. Data updated for 2026.
    Official portal: <b>www.tn.gov.in</b>
    </div>
    """, unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-title">📊 உங்கள் தகுதியான திட்டங்கள் / Your Eligible Schemes</div>', unsafe_allow_html=True)

    if submitted:
        if not user_name:
            st.error("⚠️ Please enter your name!")
        else:
            with st.spinner("🔍 AI analyzing your profile and matching 2026 TN schemes..."):
                try:
                    user_profile = f"""
Name: {user_name}
Age: {age} years
Gender: {gender}
Marital Status: {marital_status}
Annual Family Income: Rs {income}
Community Category: {category}
Occupation: {occupation}
District: {district}, Tamil Nadu
Residence Type: {residence}
Ration Card Type: {ration_card}
Education Level: {education}
Studied in Government School: {is_govt_school}
Disability Status: {has_disability}
"""
                    st.session_state.user_profile = user_profile
                    st.session_state.chat_history = []

                    vectorstore = load_schemes()
                    results = vectorstore.similarity_search(user_profile, k=10)
                    context = "\n\n".join([doc.page_content for doc in results])

                    lang_instruction = (
                        "Respond entirely in Tamil language using Tamil script."
                        if language == "Tamil"
                        else "Respond in clear, simple English."
                    )

                    client = Groq(api_key=GROQ_API_KEY)
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        max_tokens=2500,
                        messages=[
                            {
                                "role": "system",
                                "content": f"""You are an expert Tamil Nadu government scheme advisor with deep knowledge of all 2026 welfare programs.
{lang_instruction}

For EACH eligible scheme provide:
✅ Scheme Name
📋 Why eligible (based on their specific profile)
💰 Exact benefit amount
📎 Top 3 documents needed
🔗 Official apply link (use the OFFICIAL_SITE from scheme data)
📞 Helpline number

At end, briefly list schemes they are NOT eligible for and why.
Be accurate, specific, and helpful. Use emojis for readability."""
                            },
                            {
                                "role": "user",
                                "content": f"User Profile:\n{user_profile}\n\nTN Government Schemes Database 2026:\n{context}\n\nAnalyze completely and list ALL schemes this person is eligible for with full official details."
                            }
                        ]
                    )

                    result = response.choices[0].message.content
                    st.session_state.scheme_result = result
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": result
                    })
                    st.success("✅ Analysis complete!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    if st.session_state.scheme_result:
        st.markdown(f'<div class="result-box">{st.session_state.scheme_result}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">💬 Follow-up Questions / கேள்விகள் கேளுங்கள்</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history[1:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 <b>AI Advisor:</b> {msg["content"]}</div>', unsafe_allow_html=True)

        follow_up = st.text_input("Ask about any scheme...", placeholder="e.g. How to apply for Pudhumai Penn online?")

        if st.button("📨 Send / அனுப்பு", use_container_width=True) and follow_up:
            with st.spinner("Thinking..."):
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    st.session_state.chat_history.append({"role": "user", "content": follow_up})

                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert Tamil Nadu government scheme advisor for 2026. Answer clearly about eligibility, application process, documents, and official links."
                        },
                        {
                            "role": "user",
                            "content": f"User profile:\n{st.session_state.user_profile}\n\nSchemes found:\n{st.session_state.scheme_result}"
                        }
                    ] + st.session_state.chat_history

                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        max_tokens=1000,
                        messages=messages
                    )

                    reply = response.choices[0].message.content
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    st.rerun()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.markdown("""
        <div class="info-box">
        <h4>👈 எப்படி பயன்படுத்துவது / How to use:</h4>
        <ol>
        <li>உங்கள் விவரங்களை இடது பக்கம் உள்ளிடவும்</li>
        <li>"Find My Eligible Schemes" button click பண்ணவும்</li>
        <li>AI உங்களுக்கு தகுதியான அனைத்து திட்டங்களையும் காட்டும்</li>
        <li>Follow-up questions கேட்கலாம்</li>
        </ol>
        <br>
        <b>This tool covers:</b> Education, Health, Housing, Employment, Women Welfare, Senior Citizens, Sports & more!
        </div>
        """, unsafe_allow_html=True)