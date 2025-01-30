import os
os.system('pip install arabic_reshaper bidi networkx matplotlib pillow speechrecognition streamlit json PIL')
import streamlit as st
import json
import arabic_reshaper
from bidi.algorithm import get_display
import networkx as nx
import matplotlib.pyplot as plt
import speech_recognition as sr
from PIL import Image
import matplotlib.font_manager as fm

# تحميل خط يدعم العربية
arabic_font = fm.FontProperties(fname="arial.ttf")


def reshape_arabic_text(text):
    """إعادة تشكيل النص العربي ليتوافق مع الرسم"""
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)


def load_knowledge_base():
    """تحميل قاعدة المعرفة من ملف JSON"""
    file_name = "translated_diseases_v2.json"
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("symptoms", {})  # تجنب الخطأ إذا كان المفتاح غير موجود
    except FileNotFoundError:
        st.error("ملف قاعدة المعرفة غير موجود!")
        return {}


def build_semantic_network(knowledge_base):
    """بناء شبكة دلالية للأمراض والأعراض"""
    network = nx.DiGraph()
    for disease, details in knowledge_base.items():
        network.add_node(disease, type="disease")
        for symptom in details.get("الأعراض", []):
            network.add_node(symptom, type="symptom")
            network.add_edge(disease, symptom, relation="has_symptom")
    return network


def diagnose_disease(input_text, knowledge_base, semantic_network):
    """تشخيص المرض بناءً على الأعراض المدخلة"""
    input_symptoms = input_text.strip().split()
    matching_diseases = {}
    for symptom in input_symptoms:
        for disease in semantic_network.nodes:
            if semantic_network.nodes[disease].get("type") == "disease" and semantic_network.has_edge(disease, symptom):
                matching_diseases[disease] = matching_diseases.get(disease, 0) + 1
    if matching_diseases:
        best_match = max(matching_diseases, key=matching_diseases.get)
        details = knowledge_base.get(best_match, {})
        return best_match, details
    return None, None


def visualize_network(input_text, semantic_network):
    """رسم الشبكة الدلالية للأعراض المدخلة"""
    input_symptoms = input_text.strip().split()
    sub_network = nx.DiGraph()

    for symptom in input_symptoms:
        if semantic_network.has_node(symptom):
            sub_network.add_node(symptom, type="symptom")
            for disease in semantic_network.predecessors(symptom):
                sub_network.add_node(disease, type="disease")
                sub_network.add_edge(disease, symptom, relation="has_symptom")

    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(sub_network, seed=42)

    # تلوين العقد
    node_colors = ["lightblue" if sub_network.nodes[node].get("type") == "disease" else "lightcoral" for node in
                   sub_network.nodes]

    nx.draw(sub_network, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10,
            font_family="Arial")

    st.pyplot(plt)


def recognize_speech():
    """التعرف على الصوت وتحويله إلى نص"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("تحدث الآن...")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="ar")
            st.success(f"تم التعرف على النص: {text}")
            return text
        except sr.UnknownValueError:
            st.warning("لم يتم التعرف على أي صوت. حاول مرة أخرى.")
        except sr.RequestError:
            st.error("حدث خطأ في الاتصال بخدمة التعرف على الصوت.")
    return ""


def main():
    st.title("نظام تشخيص طبي")

    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return

    semantic_network = build_semantic_network(knowledge_base)

    image = Image.open("images (2).jpg")
    st.image(image, use_container_width=True)

    # إدخال نص الأعراض يدويًا
    input_text = st.text_input("أدخل الأعراض:")

    if st.button("تشخيص باستخدام الميكروفون"):
        input_text = recognize_speech()
        if input_text:
            disease, details = diagnose_disease(input_text, knowledge_base, semantic_network)
            if disease:
                st.success(f"🦠 المرض المحتمل: {disease}")
                st.write(f"📖 الوصف: {details.get('الوصف', '')}")
                st.write(f"💊 العلاج: {', '.join(details.get('العلاج', []))}")
            else:
                st.warning("⚠️ لم يتم العثور على مرض مطابق.")

    # زر التشخيص
    if st.button("تشخيص"):
        if input_text:
            disease, details = diagnose_disease(input_text, knowledge_base, semantic_network)
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
    if st.button("عرض الشبكة"):
        if input_text:
            visualize_network(input_text, semantic_network)
        else:
            st.warning("الرجاء إدخال الأعراض أولاً.")


if __name__ == "__main__":
    main()
