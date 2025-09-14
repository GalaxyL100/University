# ایمپورت کردن کتابخانه‌های مورد نیاز
import json
import sys
import heapq
import pandas as pd
from difflib import get_close_matches
from langdetect import detect

# کتابخانه‌های رابط کاربری گرافیکی (PyQt6)
from PyQt6.QtWidgets import (
    QApplication,
    QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QListWidget, QTabWidget,
    QListWidgetItem, QMessageBox, QTextEdit, QInputDialog
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

# کتابخانه‌های زبان فارسی (Hazm)
from hazm import Normalizer, SentenceTokenizer, WordTokenizer

# کتابخانه خلاصه‌سازی متن انگلیسی (Sumy)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# بارگذاری دیتاست احساسات
df = pd.read_csv(r'C:\Users\pc\Documents\university file\final project\emotion_dataset.csv')


# تعریف کلاس چت‌بات
class ChatBot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    # بارگذاری داده‌های سوال و جواب از فایل JSON
    def load_data(self):
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"questions": []}

    # ذخیره داده‌های یادگرفته‌شده در فایل
    def save_data(self):
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    # پاسخ دادن به سوال کاربر با تطبیق سوالات قبلی
    def get_response(self, user_input: str) -> str:
        if not self.data or 'questions' not in self.data:
            return None
    
        user_input_lower = user_input.strip().lower()
    
        # جستجوی دقیق
        for q in self.data['questions']:
            if q['question'].strip().lower() == user_input_lower:
                return q['answer']
    
        # جستجوی جزئی 
        for q in self.data['questions']:
            if user_input_lower in q['question'].strip().lower():
                return q['answer']
            
        for q in self.data['questions']:

            question_words = q['question'].strip().lower().split()
            user_words = user_input.strip().lower().split()

            common_words = set(question_words) & set(user_words)
            common_count = len(common_words)
            if common_count >= 3:
                return q['answer']
           
    # یادگیری پاسخ جدید در صورت بلد نبودن
    def learn_new_answer(self, question: str, answer: str):
        self.data['questions'].append({'question': question, 'answer': answer})
        self.save_data()


# تابع خلاصه‌سازی متن فارسی با استفاده از توکن‌سازی و وزن‌دهی به جملات
def summarize_farsi(text):
    normalizer = Normalizer()
    text = normalizer.normalize(text)

    sent_tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()

    sentences = sent_tokenizer.tokenize(text)
    word_freq = {}

    for sent in sentences:
        words = word_tokenizer.tokenize(sent)
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenizer.tokenize(sent):
            if word in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)


# تابع خلاصه‌سازی متن انگلیسی با الگوریتم TextRank
def summarize_english(text):
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)
    return " ".join([str(sentence) for sentence in summary])


# رابط کاربری برنامه با استفاده از PyQt6
class ChatUI(QWidget):
    def __init__(self, bot: ChatBot):
        super().__init__()
        self.bot = bot
        self.setWindowTitle("UnderFeel")
        self.setMinimumSize(500, 600)
        self.selected_theme = self.load_last_theme()
        self.init_ui()
        self.apply_theme(self.selected_theme)

    # بارگذاری آخرین تم
    def load_last_theme(self):
        try:
            with open("settings.json", "r", encoding="utf-8") as f:
                settings = json.load(f)
                return settings.get("theme", "pink")
        except (FileNotFoundError, json.JSONDecodeError):
            return "pink"

    # تغییر تم
    def change_theme(self, theme_name):
        self.selected_theme = theme_name
        self.apply_theme(theme_name)

    # ذخیره تم با تایید کاربر
    def save_theme_with_confirm(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("تایید ذخیره تم")
        msg_box.setText(f"آیا می‌خواهید تم رنگی '{self.selected_theme}' ذخیره شود؟")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel)
        ret = msg_box.exec()

        if ret == QMessageBox.StandardButton.Save:
            settings = {"theme": self.selected_theme}
            with open("settings.json", "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "ذخیره شد", f"تم '{self.selected_theme}' ذخیره شد!")
      
    # تابع تغییر تم‌ها
    def apply_theme(self, theme_name):
        themes = {
            "yellow": """
                QWidget { background-color: #fff9db; 
                font-family: 'Vazirmatn'; }
                QLineEdit, QTextEdit { background-color: #fff3bf;
                border: 2px solid #ffd43b;
                border-radius: 10px; 
                padding: 8px; 
                font-size: 14px; }
                QPushButton { background-color: #fcc419; 
                color: black; border: none; 
                padding: 8px 16px; 
                border-radius: 10px; 
                font-weight: bold; }
                QPushButton:hover { background-color: #fab005; }
                QListWidget { background-color: #fff9db; border: none; padding: 10px; border-radius: 10px; }
                QLabel#header { color: #FDB913; }
                QLabel#header_tab4 { color: #FDB913; }
            """,
            "green": """
                QWidget { background-color: #ccffcc; 
                font-family: 'Vazirmatn'; }
                QLineEdit, QTextEdit { background-color: #99cc99; 
                border: 2px solid #4b6d4b; border-radius: 10px; 
                padding: 8px; font-size: 14px; }
                QPushButton { background-color: #669966; 
                color: white; border: none; 
                padding: 8px 16px; 
                border-radius: 10px; 
                font-weight: bold; }
                QPushButton:hover { background-color: #4b6d4b; }
                QListWidget { background-color: #ccffcc; 
                border: none; 
                padding: 10px; 
                border-radius: 10px; }
                QLabel#header { color: #4b6d4b; }
                QLabel#header_tab4 { color: #4b6d4b; }
            """,
            "blue": """
                QWidget { background-color: #e7f5ff; 
                font-family: 'Vazirmatn'; }
                QLineEdit, QTextEdit { background-color: #d0ebff; 
                border: 2px solid #339af0; 
                border-radius: 10px; 
                padding: 8px; 
                font-size: 14px; }
                QPushButton { background-color: #339af0; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 10px; 
                font-weight: bold; }
                QPushButton:hover { background-color: #1c7ed6; }
                QListWidget { background-color: #e7f5ff; 
                border: none; padding: 10px; 
                border-radius: 10px; }
                QLabel#header { color: #1c7ed6; }
                QLabel#header_tab4 { color: #1c6ed6; }
            """,
            "black": """
                QWidget { background-color: #1e1e1e; 
                font-family: 'Vazirmatn'; 
                color: white; }
                QLineEdit, QTextEdit { background-color: #2c2c2c; 
                border: 2px solid #555; 
                border-radius: 10px; 
                padding: 8px; 
                font-size: 14px; 
                color: white; }
                QPushButton { background-color: #444; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 10px; 
                font-weight: bold; }
                QPushButton:hover { background-color: #666; }
                QListWidget { background-color: #2c2c2c; 
                border: none; 
                padding: 10px; 
                border-radius: 10px; 
                color: white; }
                QLabel#header { color: #666; }
                QLabel#header_tab4 { color: #666; }
                
            """,
            "white": """
                QWidget { background-color: #ffffff; 
                font-family: 'Vazirmatn'; }
                QLineEdit, QTextEdit { background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 10px; 
                padding: 8px; 
                font-size: 14px; }
                QPushButton { background-color: #adb5bd; 
                color: black; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 10px; 
                font-weight: bold; }
                QPushButton:hover { background-color: #868e96; }
                QListWidget { background-color: #f8f9fa; 
                border: none; 
                padding: 10px; 
                border-radius: 10px; }
                QLabel#header { color: #868e96; }
                QLabel#header_tab4 { color: #868e96; }

            """,
            "pink": """
                QWidget { background-color: #ffe0f0; 
                font-family: 'Vazirmatn'; }
                QLineEdit, QTextEdit { background-color: #ffd6e7;
                border: 2px solid #ff80ab;
                border-radius: 10px; 
                padding: 8px; 
                font-size: 14px; }
                QPushButton { background-color: #ff80ab; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 10px; 
                font-weight: bold; }
                QPushButton:hover { background-color: #ff4081; }
                QListWidget { background-color: #ffe0f0; 
                border: none; padding: 10px; border-radius: 10px; }
                QLabel#header { color: #d63384; }
                QLabel#header_tab4 { color: #d63384; }
"""
        }

        if theme_name in themes:
            self.setStyleSheet(themes[theme_name])
        

    # ساخت رابط کاربری اصلی شامل سه تب
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # تب اول: چت‌بات
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        self.header = QLabel("💬 چت با UnderFeel Bot")
        self.header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        #header.setStyleSheet("color: #d63384; margin-bottom: 10px;")
        #tab1_layout.addWidget(header)
        tab1_layout.addWidget(self.header)
        self.header.setObjectName("header")

        self.chat_area = QListWidget()
        tab1_layout.addWidget(self.chat_area)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.send_button = QPushButton("ارسال")
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        tab1_layout.addLayout(input_layout)
        self.send_button.clicked.connect(self.handle_user_input)

        # تب دوم: تحلیل احساسات
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        label2 = QLabel("🔍 یک جمله بنویس تا احساسشو بفهمیم:")
        tab2_layout.addWidget(label2)

        self.sentiment_input = QTextEdit()
        tab2_layout.addWidget(self.sentiment_input)

        self.sentiment_button = QPushButton("احساساتمو تحلیل کن")
        tab2_layout.addWidget(self.sentiment_button)

        self.sentiment_output = QLabel("✉️ نتیجه نمایش داده می‌شود...")
        self.sentiment_output.setStyleSheet("font-weight: bold; color: #8b008b;")
        tab2_layout.addWidget(self.sentiment_output)

        self.sentiment_button.clicked.connect(self.analyze_sentiment)

        # تب سوم: خلاصه‌ساز متن
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        label3 = QLabel("📝 لطفاً متن فارسی یا انگلیسی را وارد کنید:")
        tab3_layout.addWidget(label3)

        self.text_input = QTextEdit()
        tab3_layout.addWidget(self.text_input)

        self.summarize_button = QPushButton("خلاصه کن")
        tab3_layout.addWidget(self.summarize_button)

        self.result_label = QLabel("✅ خلاصه متن:")
        tab3_layout.addWidget(self.result_label)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)
        tab3_layout.addWidget(self.result_output)

        self.summarize_button.clicked.connect(self.summarize_text)

# تب چهارم: تنظیمات تم
        tab4 = QWidget()
        tab4_layout = QVBoxLayout(tab4)

# هدر تب چهارم
        self.header_tab4 = QLabel("🎨 انتخاب تم رنگی")
        self.header_tab4.setObjectName("header_tab4")  # برای CSS جدا
        self.header_tab4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.header_tab4.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        tab4_layout.addWidget(self.header_tab4)
        tab4_layout.setAlignment(self.header_tab4, Qt.AlignmentFlag.AlignTop)

# دکمه‌های انتخاب تم با رنگ مخصوص
        themes_buttons = [
            ("تم زرد", "yellow", "#fcc419"),
            ("تم سبز", "green", "#38d9a9"),
            ("تم آبی", "blue", "#339af0"),
            ("تم مشکی", "black", "#444"),
            ("تم سفید", "white", "#e1e3e6"),
            ("تم صورتی", "pink", "#ff80ab"),
]

        for text, theme, color in themes_buttons:
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, t=theme: self.change_theme(t))
            text_color = "black" if theme == "white" else "white"
            btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: {text_color};
                border: none;
                padding: 10px 18px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 14px;
    }}
        QPushButton:hover {{
                background-color: {color};
                opacity: 0.85;
    }}
        """)

            tab4_layout.addWidget(btn)   # حتماً داخل حلقه باشه ✅

# دکمه ذخیره تم
        save_btn = QPushButton("💾 ذخیره تم")
        save_btn.clicked.connect(self.save_theme_with_confirm)
        tab4_layout.addWidget(save_btn)

# اضافه کردن تب چهارم به تب‌ها
        # اضافه کردن تب‌ها
        tabs.addTab(tab1, "چت‌بات")
        tabs.addTab(tab2, "تحلیل احساسات")
        tabs.addTab(tab3, "خلاصه‌ساز")
        tabs.addTab(tab4, "تنظیمات")

# اضافه کردن کل تب‌ها به صفحه اصلی
        main_layout.addWidget(tabs)



    # تابع ارسال پیام در چت‌بات
    def handle_user_input(self):
        user_text = self.input_field.text().strip()
        if not user_text:
            return

        self.add_chat_message(f"🧑‍💻 شما: {user_text}", Qt.AlignmentFlag.AlignRight)
        self.input_field.clear()

        response = self.bot.get_response(user_text)
        if response:
            self.add_chat_message(f"🤖 بات: {response}", Qt.AlignmentFlag.AlignLeft)
        else:
            self.add_chat_message("🤖 بات: جواب اینو بلد نیستم، لطفا یادم بده.", Qt.AlignmentFlag.AlignLeft)
            self.ask_to_learn(user_text)

    # اضافه کردن پیام به رابط چت
    def add_chat_message(self, text, align):
        item = QListWidgetItem(text)
        item.setTextAlignment(align)
        self.chat_area.addItem(item)

    # دریافت پاسخ مناسب از کاربر در صورت بلد نبودن بات
    def ask_to_learn(self, question):
        answer, ok = QInputDialog.getText(self, "یادگیری", f"پاسخ مناسب برای '{question}' چیه؟")
        if ok and answer.strip():
            self.bot.learn_new_answer(question, answer.strip())
            self.add_chat_message("🤖 بات: ممنون! یاد گرفتم.", Qt.AlignmentFlag.AlignLeft)

    # خلاصه‌سازی متن بر اساس زبان شناسایی‌شده
    def summarize_text(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.result_output.setPlainText("⛔️ لطفاً متنی وارد کنید.")
            return

        try:
            lang = detect(text)
            if lang == 'fa':
                summary = summarize_farsi(text)
                self.result_output.setPlainText("📌 زبان: فارسی\n\n" + summary)
            elif lang == 'en':
                summary = summarize_english(text)
                self.result_output.setPlainText("📌 Language: English\n\n" + summary)
            else:
                self.result_output.setPlainText("⛔️ زبان پشتیبانی نمی‌شود.")
        except Exception as e:
            self.result_output.setPlainText(f"⚠️ خطا در پردازش: {e}")

    # تحلیل احساسات با استفاده از کلمات موجود در دیتاست
    def analyze_sentiment(self):
        text = self.sentiment_input.toPlainText().strip()
        if not text:
            self.sentiment_output.setText("⛔️ لطفاً جمله‌ای وارد کنید.")
            return

        # دسته‌بندی کلمات احساسات از دیتاست
        uneasy_words = df[df['emotion'] == 'مضطرب']['word'].tolist()
        envy_words = df[df['emotion'] == 'حسادت']['word'].tolist()
        angry_words = df[df['emotion'] == 'عصبانی']['word'].tolist()
        love_words = df[df['emotion'] == 'عشق']['word'].tolist()
        happy_words = df[df['emotion'] == 'خوشحالی']['word'].tolist()
        sad_words = df[df['emotion'] == 'ناراحت']['word'].tolist()
        hate_words = df[df['emotion'] == 'نفرت']['word'].tolist()

        scores = {
            'مضطرب': 0, 'حسادت': 0, 'عصبانی': 0, 'عشق': 0,
            'خوشحالی': 0, 'ناراحت': 0, 'نفرت': 0
        }

        # نرمال‌سازی و توکن‌سازی جمله
        normalizer = Normalizer()
        tokenizer = WordTokenizer()
        text = normalizer.normalize(text)
        words = tokenizer.tokenize(text)

        # شمارش کلمات احساس‌برانگیز
        for word in words:
            if word in uneasy_words:
                scores['مضطرب'] += 1
            elif word in envy_words:
                scores['حسادت'] += 1
            elif word in angry_words:
                scores['عصبانی'] += 1
            elif word in love_words:
                scores['عشق'] += 1
            elif word in happy_words:
                scores['خوشحالی'] += 1
            elif word in sad_words:
                scores['ناراحت'] += 1
            elif word in hate_words:
                scores['نفرت'] += 1

        # نمایش احساس غالب
        if all(score == 0 for score in scores.values()):
            self.sentiment_output.setText("🤔 هیچ احساسی شناسایی نشد.")
        else:
            dominant_emotion = max(scores, key=scores.get)
            self.sentiment_output.setText(f"😊 احساس جمله {dominant_emotion} بود.")


# اجرای برنامه
if __name__ == "__main__":
    app = QApplication(sys.argv)
    bot = ChatBot(r"C:\Users\pc\Documents\university file\final project\data.json")
    window = ChatUI(bot)
    window.show()
    sys.exit(app.exec())




