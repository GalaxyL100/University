# ایمپورت کردن کتابخانه‌های مورد نیاز
import json
import sys
import heapq
import pandas as pd
from difflib import get_close_matches
from langdetect import detect

# کتابخانه‌های رابط کاربری گرافیکی (PyQt6)
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
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
        questions = [q['question'] for q in self.data['questions']]
        best_match = get_close_matches(user_input, questions, n=1, cutoff=0.6)

        if best_match:
            for q in self.data['questions']:
                if q['question'] == best_match[0]:
                    return q['answer']
        return None

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
        self.setWindowTitle("UnderFeel")  # عنوان پنجره
        self.setMinimumSize(500, 600)     # اندازه حداقل پنجره

        # اعمال استایل گرافیکی با CSS داخلی
        self.setStyleSheet("""
            QWidget {
                background-color: #ffe4ec;
                font-family: 'Vazirmatn', sans-serif;
            }
            QLineEdit, QTextEdit {
                background-color: #fff0f5;
                border: 2px solid #ffb6c1;
                border-radius: 10px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #ff69b4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff1493;
            }
            QListWidget {
                background-color: #ffeef6;
                border: none;
                padding: 10px;
                border-radius: 10px;
            }
        """)
        self.init_ui()

    # ساخت رابط کاربری اصلی شامل سه تب
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # تب اول: چت‌بات
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        header = QLabel("💬 چت با UnderFeel Bot")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setStyleSheet("color: #d63384; margin-bottom: 10px;")
        tab1_layout.addWidget(header)

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

        # اضافه کردن تب‌ها به صفحه اصلی
        tabs.addTab(tab1, "چت بات")
        tabs.addTab(tab2, "تحلیل احساس")
        tabs.addTab(tab3, "خلاصه‌ساز")
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
