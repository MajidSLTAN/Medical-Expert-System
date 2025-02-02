import streamlit as st
import json
import arabic_reshaper
from bidi.algorithm import get_display
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.font_manager as fm
import re
from collections import Counter
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr

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
            return data.get("symptoms", {})
    except FileNotFoundError:
        st.error("⚠️ ملف قاعدة المعرفة غير موجود!")
        return {}


def preprocess_text(text):
    """تنظيف وتحليل النص المدخل"""
    text = re.sub(r'[^\w\s]', '', text)  # إزالة الرموز الخاصة
    text = re.sub(r'إ', "ا", text)  # تصحيح بعض الحروف
    text = re.sub(r'أ', "ا", text)
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


def diagnose_disease(input_text, knowledge_base, semantic_network):
    """تشخيص المرض بناءً على الأعراض المدخلة"""
    input_symptoms = preprocess_text(input_text)
    disease_scores = Counter()

    for disease in knowledge_base:
        matched_symptoms = sum(1 for symptom in input_symptoms if symptom in knowledge_base[disease]["الأعراض"])
        if matched_symptoms:
            disease_scores[disease] = matched_symptoms

    if disease_scores:
        best_match = disease_scores.most_common(1)[0][0]
        details = knowledge_base.get(best_match, {})
        return best_match, details
    return None, None


def visualize_network(input_text, semantic_network):
    """رسم الشبكة الدلالية للأعراض المدخلة"""
    input_symptoms = preprocess_text(input_text)
    sub_network = nx.DiGraph()

    for symptom in input_symptoms:
        if symptom in semantic_network.nodes:
            sub_network.add_node(symptom, type="symptom")
            for disease in semantic_network.predecessors(symptom):
                sub_network.add_node(disease, type="disease")
                sub_network.add_edge(disease, symptom, relation="has_symptom")

    if sub_network.nodes:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(sub_network, seed=42)
        node_colors = ["lightblue" if sub_network.nodes[node].get("type") == "disease" else "lightcoral" for node in
                       sub_network.nodes]
        nx.draw(sub_network, pos, with_labels=True, node_color=node_colors, node_size=1500, font_size=10,
                font_family="Arial")
        st.pyplot(plt)
    else:
        st.warning("⚠️ لم يتم العثور على شبكة دلالية للأعراض المدخلة.")


def recognize_speech():
    """التعرف على الصوت وتحويله إلى نص باستخدام sounddevice"""
    fs = 16000  # معدل العينة (16 kHz)
    duration = 5  # مدة التسجيل بالثواني
    st.info("🎤 تحدث الآن...")

    try:
        # تسجيل الصوت
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
        sd.wait()  # انتظر حتى ينتهي التسجيل
        st.success("✅ تم تسجيل الصوت بنجاح!")

        # حفظ التسجيل كملف مؤقت
        write("temp_recording.wav", fs, recording)

        # تحويل الصوت إلى نص باستخدام Google Speech Recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp_recording.wav") as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language="ar")
            return text
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء تسجيل الصوت: {e}")
        return ""


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
    if st.button("📊عرض الشبكة"):
        if input_text:
            visualize_network(input_text, semantic_network)
        else:
            st.warning("الرجاء إدخال الأعراض أولاً.")


if __name__ == "__main__":
    main()
