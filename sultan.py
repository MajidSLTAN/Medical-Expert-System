import streamlit as st
import json
import arabic_reshaper
from bidi.algorithm import get_display
import networkx as nx
import matplotlib.pyplot as plt
import speech_recognition as sr
from PIL import Image
import matplotlib.font_manager as fm
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gdown
# تحميل خط يدعم العربية
arabic_font = fm.FontProperties(fname="arial.ttf")


def reshape_arabic_text(text):
    """إعادة تشكيل النص العربي ليتوافق مع الرسم"""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


@st.cache_data
def load_knowledge_base():
    """تحميل قاعدة المعرفة من ملف JSON"""
    file_id = "1_Yi0K6hVGDJjZRhoxBJxrwyKLxbRPlCc"  # Replace with your file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "translated_diseases_v2.json"  # Output file name
    gdown.download(url, output, quiet=False, fuzzy=True)  # Use fuzzy to handle shareable links

    try:
        with open(output, "r", encoding="utf-8") as file:
            content = file.read()

            if not content:
                st.error("⚠️ ملف قاعدة المعرفة فارغ!")
                return {}
            data = json.loads(content)
            return data.get("symptoms", {})
    except FileNotFoundError:
        st.error("⚠️ ملف قاعدة المعرفة غير موجود!")
        return {}
    except json.JSONDecodeError:
        st.error("⚠️ ملف قاعدة المعرفة غير صالح!")
        return {}


def preprocess_text(text):
    """تنظيف وتحليل النص المدخل"""
    text = re.sub(r'[^\w\s]', '', text)  # إزالة الرموز الخاصة
    text = re.sub(r'[أإآ]', "ا", text)  # تطبيع الحروف
    text = re.sub(r'ة', "ه", text)  # تطبيع الحروف
    return text.strip().split()


def build_semantic_network(knowledge_base):
    """بناء شبكة دلالية للأمراض والأعراض"""
    network = nx.DiGraph()
    for disease, details in knowledge_base.items():
        network.add_node(disease, type="disease")
        for symptom in details.get("الأعراض", []):
            network.add_node(symptom, type="symptom")
            network.add_edge(disease, symptom, relation="has_symptom")
    return network


def calculate_similarity(input_symptoms, disease_symptoms):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([input_symptoms, disease_symptoms])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]
def diagnose_disease(input_text, knowledge_base, semantic_network):
    """تشخيص المرض بناءً على الأعراض المدخلة مع تحسين دقة المطابقة"""
    input_symptoms = preprocess_text(input_text)
    disease_scores = []

    for disease in semantic_network.nodes:
        if semantic_network.nodes[disease].get("type") == "disease":
            disease_symptoms = [symptom for symptom in semantic_network.successors(disease)]

            # حساب التشابه الدلالي
            input_symptoms_str = " ".join(input_symptoms)
            disease_symptoms_str = " ".join(disease_symptoms)
            similarity = calculate_similarity(input_symptoms_str, disease_symptoms_str)

            # حساب عدد الأعراض المتطابقة
            matched_symptoms = sum(1 for symptom in input_symptoms if symptom in disease_symptoms)
            total_symptoms = len(disease_symptoms)

            if matched_symptoms > 0:
                match_ratio = matched_symptoms / total_symptoms
                disease_scores.append((disease, match_ratio, matched_symptoms, similarity))

    if disease_scores:
        # ترتيب الأمراض حسب التشابه الدلالي ثم نسبة المطابقة
        disease_scores.sort(key=lambda x: (-x[3], -x[1], -x[2]))
        best_match = disease_scores[0][0]
        details = knowledge_base.get(best_match, {})
        return best_match, details, disease_scores
    return None, None, []


def visualize_network(input_text, semantic_network):
    """رسم الشبكة الدلالية للأعراض المدخلة"""
    input_symptoms = preprocess_text(input_text)
    sub_network = nx.DiGraph()

    for symptom in input_symptoms:
        if semantic_network.has_node(symptom):
            sub_network.add_node(symptom, type="symptom")
            for disease in semantic_network.predecessors(symptom):
                sub_network.add_node(disease, type="disease")
                sub_network.add_edge(disease, symptom, relation="has_symptom")

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(sub_network, seed=42)
    node_colors = ["lightblue" if sub_network.nodes[node].get("type") == "disease" else "lightcoral" for node in
                   sub_network.nodes]

    nx.draw(sub_network, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10,
            font_family="Arial")
    st.pyplot(plt)

def main():
    st.title("🔬 نظام تشخيص طبي ذكي")

    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return

    semantic_network = build_semantic_network(knowledge_base)

    image = Image.open("images (2).jpg")
    st.image(image, use_container_width=True)

    # إدخال نص الأعراض يدويًا
    input_text = st.text_input("أدخل الأعراض:")

    # زر التشخيص
    if st.button("تشخيص"):
        if input_text:
            disease, details, ranked_diseases = diagnose_disease(input_text, knowledge_base, semantic_network)
            if disease:
                st.success(f"🦠 المرض المحتمل: {disease}")
                st.write(f"📖 الوصف: {details.get('الوصف', '')}")
                st.write(f"💊 العلاج: {', '.join(details.get('العلاج', []))}")
            else:
                st.warning("⚠️ لم يتم العثور على مرض مطابق.")
    # إدخال اسم المرض للبحث العكسي
    reverse_search = st.text_input("🔍 أدخل اسم المرض للبحث عن أعراضه:")

    if st.button("🔄 البحث العكسي"):
        if reverse_search:
            disease_name = reverse_search.strip()
            if disease_name in knowledge_base:
                details = knowledge_base[disease_name]
                symptoms = details.get("الأعراض", [])
                st.success(f"✅ الأعراض المرتبطة بـ {disease_name}:")
                st.write(f"📝 الأعراض: {', '.join(symptoms)}")
            else:
                st.warning("⚠️ لم يتم العثور على هذا المرض في قاعدة البيانات.")
        else:
            st.warning("❌ الرجاء إدخال اسم المرض للبحث عنه.")

    # زر عرض الشبكة
    if st.button("📊عرض الشبكة"):
        if input_text:
            visualize_network(input_text, semantic_network)
        else:
            st.warning("الرجاء إدخال الأعراض أولاً.")


if __name__ == "__main__":
    main()
