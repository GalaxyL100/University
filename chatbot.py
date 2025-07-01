import pandas as pd
import json
import sys
import heapq
from difflib import get_close_matches
from langdetect import detect

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QListWidget, QTabWidget,
    QListWidgetItem, QMessageBox, QTextEdit, QInputDialog
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from hazm import Normalizer, SentenceTokenizer, WordTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

class ChatBot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()

    def load_data(self):
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"questions": []}

    def save_data(self):
        with open(self.data_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def get_response(self, user_input: str) -> str:
        questions = [q['question'] for q in self.data['questions']]
        best_match = get_close_matches(user_input, questions, n=1, cutoff=0.6)

        if best_match:
            for q in self.data['questions']:
                if q['question'] == best_match[0]:
                    return q['answer']
        return None

    def learn_new_answer(self, question: str, answer: str):
        self.data['questions'].append({'question': question, 'answer': answer})
        self.save_data()

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

def summarize_english(text):
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)
    return " ".join([str(sentence) for sentence in summary])

# ... تمام importها مثل قبل

# در ادامه بقیه‌ی کد مثل قبل بدون تغییر، تا می‌رسیم به تعریف کلاس ChatUI

class ChatUI(QWidget):
    def __init__(self, bot: ChatBot):
        super().__init__()
        self.bot = bot
        self.setWindowTitle("UnderFeel")
        self.setMinimumSize(500, 600)
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

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)

        # --- سربرگ اول: چت‌بات ---
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

        # --- سربرگ دوم: تحلیل احساس ---
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

        # --- سربرگ سوم: خلاصه‌ساز ---
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

        # افزودن تب‌ها
        tabs.addTab(tab1, "چت بات")
        tabs.addTab(tab2, "تحلیل احساس")
        tabs.addTab(tab3, "خلاصه‌ساز")
        main_layout.addWidget(tabs)

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

    def add_chat_message(self, text, align):
        item = QListWidgetItem(text)
        item.setTextAlignment(align)
        self.chat_area.addItem(item)

    def ask_to_learn(self, question):
        answer, ok = QInputDialog.getText(self, "یادگیری", f"پاسخ مناسب برای '{question}' چیه؟")
        if ok and answer.strip():
            self.bot.learn_new_answer(question, answer.strip())
            self.add_chat_message("🤖 بات: ممنون! یاد گرفتم.", Qt.AlignmentFlag.AlignLeft)

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

    def analyze_sentiment(self):
        text = self.sentiment_input.toPlainText().strip()
        if not text:
            self.sentiment_output.setText("⛔️ لطفاً جمله‌ای وارد کنید.")
            return

        # واژگان نمونه ساده برای تحلیل احساس
        positive_words = ['عالی', 'خوشحال', 'شاد', 'موفق', 'زیبا', 'دلنشین', 'عاشق', 'دوست‌داشتنی']
        negative_words = ['بد', 'ناراحت', 'غمگین', 'افتضاح', 'تنفر', 'خسته', 'ضعیف', 'عصبانی']

        normalizer = Normalizer()
        tokenizer = WordTokenizer()
        text = normalizer.normalize(text)
        words = tokenizer.tokenize(text)

        pos_score = sum(word in positive_words for word in words)
        neg_score = sum(word in negative_words for word in words)

        if pos_score > neg_score:
            self.sentiment_output.setText("😊 احساس جمله مثبت بود.")
        elif neg_score > pos_score:
            self.sentiment_output.setText("😞 احساس جمله منفی بود.")
        else:
            self.sentiment_output.setText("😐 احساس خاصی تشخیص داده نشد (خنثی).")

# --------------------------
# اجرای برنامه
# --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    bot = ChatBot(r"C:\Users\pc\Documents\university file\final project\data.json")
    window = ChatUI(bot)
    window.show()
    sys.exit(app.exec())
