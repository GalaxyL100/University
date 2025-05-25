import json
import sys
from difflib import get_close_matches
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QListWidget, QTabWidget,
    QListWidgetItem, QMessageBox
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


# --------------------------
# بخش ۱: کلاس چت‌بات
# --------------------------
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


# --------------------------
# بخش ۲: رابط گرافیکی
# --------------------------
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
            QLineEdit {
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

        # تب اول (چت بات)
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

        # تب دوم
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.addWidget(QLabel("<!-- اینجا کدهاتو واسه سربرگ دوم وارد کن -->"))

        # تب سوم
        tab3 = QWidget()
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.addWidget(QLabel("<!-- اینجا کدهاتو واسه سربرگ سوم وارد کن -->"))

        # اضافه کردن تب‌ها
        tabs.addTab(tab1, "سربرگ ۱")
        tabs.addTab(tab2, "سربرگ ۲")
        tabs.addTab(tab3, "سربرگ ۳")
        main_layout.addWidget(tabs)

    def handle_user_input(self):
        user_text = self.input_field.text().strip().lower()
        if not user_text:
            return

        self.add_chat_message(f"🧑‍💻 شما: {user_text}", align=Qt.AlignmentFlag.AlignRight)
        self.input_field.clear()

        response = self.bot.get_response(user_text)

        if response:
            self.add_chat_message(f"🤖 بات: {response}", align=Qt.AlignmentFlag.AlignLeft)
        else:
            self.add_chat_message("🤖 بات: جواب اینو بلد نیستم، لطفا بهم یاد بده.", align=Qt.AlignmentFlag.AlignLeft)
            self.ask_to_learn(user_text)

    def add_chat_message(self, text, align):
        item = QListWidgetItem(text)
        item.setTextAlignment(align)
        self.chat_area.addItem(item)

    def ask_to_learn(self, question):
        answer, ok = QMessageBox.getText(self, "یادگیری", f"پاسخ مناسب برای '{question}' چیه؟")
        if ok and answer.strip():
            self.bot.learn_new_answer(question, answer.strip())
            self.add_chat_message("🤖 بات: ممنون! یاد گرفتم.", align=Qt.AlignmentFlag.AlignLeft)


# --------------------------
# بخش ۳: اجرای برنامه
# --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    bot = ChatBot(r"C:\Users\pc\documents\university file\final project - Copy\chat_data.json")
    window = ChatUI(bot)
    window.show()
    sys.exit(app.exec())
